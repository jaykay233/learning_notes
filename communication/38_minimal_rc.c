/*
 * 38_minimal_rc.c —— 一个完整的 RC SEND/RECV 程序（服务器 + 客户端，同一个二进制）
 *
 * 编译（需要 rdma-core 开发包，Linux）：
 *   gcc -O2 -Wall -o minimal_rc 38_minimal_rc.c -libverbs -lrdmacm
 *
 * 运行：
 *   服务器： ./minimal_rc server 7471
 *   客户端： ./minimal_rc client 192.168.1.10 7471
 *
 * 它做的事：客户端发一条 64 字节消息，服务器预先投好一个 recv WR 并打印收到的内容。
 * 消息大小和条数都写在下面两个宏里，改完重编即可。
 *
 * 这份代码故意把每一种资源都显式写出来，不做封装。
 * 目的不是"写得漂亮"，而是让 ibv_* 的调用顺序、每个对象的依赖关系一眼可见。
 * 每个阶段的注释对应 38 章正文的小节号。
 */

#define _GNU_SOURCE
#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <infiniband/verbs.h>
#include <rdma/rdma_cma.h>

#define MSG_LEN 64          /* 一条消息的字节数 */
#define BUFFER_LEN 4096     /* 注册的 MR 大小，必须 >= MSG_LEN */

/* ------------------------------------------------------------------ 工具 */

static void die(const char *what)
{
	fprintf(stderr, "%s: %s\n", what, strerror(errno));
	exit(EXIT_FAILURE);
}

static void check(int rc, const char *what)
{
	if (rc) {
		fprintf(stderr, "%s: %s (rc=%d)\n", what, strerror(rc), rc);
		exit(EXIT_FAILURE);
	}
}

static const char *wc_status_name(enum ibv_wc_status s)
{
	/* 只列正文里讲到的几种，方便对照 */
	switch (s) {
	case IBV_WC_SUCCESS:            return "IBV_WC_SUCCESS";
	case IBV_WC_LOC_LEN_ERR:        return "IBV_WC_LOC_LEN_ERR";
	case IBV_WC_LOC_PROT_ERR:       return "IBV_WC_LOC_PROT_ERR";
	case IBV_WC_LOC_ACCESS_ERR:     return "IBV_WC_LOC_ACCESS_ERR";
	case IBV_WC_REM_ACCESS_ERR:     return "IBV_WC_REM_ACCESS_ERR";
	case IBV_WC_REM_INV_REQ_ERR:    return "IBV_WC_REM_INV_REQ_ERR";
	case IBV_WC_RETRY_EXC_ERR:      return "IBV_WC_RETRY_EXC_ERR";
	case IBV_WC_RNR_RETRY_EXC_ERR:  return "IBV_WC_RNR_RETRY_EXC_ERR";
	case IBV_WC_WR_FLUSH_ERR:       return "IBV_WC_WR_FLUSH_ERR";
	default:                        return "other";
	}
}

/* ------------------------------------------------------------------ 资源 */

/*
 * §3 资源集合：PD / MR / CQ / QP 都挂在这个结构上。
 * 真实项目里它们通常是长生命周期的，这里为了短程序放成一坨。
 */
struct resources {
	struct ibv_pd  *pd;
	struct ibv_mr  *mr;
	struct ibv_cq  *cq;
	struct ibv_qp  *qp;
	char           *buf;      /* 实际收发用的内存 */
};

/* §4 一步到位：建 PD -> 注册 MR -> 建 CQ -> 建 QP -> 改到 RTS */
static void build_resources(struct rdma_cm_id *id, struct resources *res)
{
	struct ibv_qp_init_attr qp_attr;
	struct ibv_qp_attr attr;
	int rc;

	res->pd = ibv_alloc_pd(id->verbs);
	if (!res->pd)
		die("ibv_alloc_pd");

	/*
	 * §5.1 注册 MR。三个 access flag 缺一不可：
	 *   LOCAL_WRITE   —— post_recv 的缓冲区必须可写
	 *   REMOTE_WRITE  —— 允许对端 WRITE 进来（本例用 SEND，其实用不到，
	 *                    但保留它，方便你把它改成单边写做实验）
	 *   REMOTE_READ   —— 允许对端 READ 出去
	 */
	res->buf = calloc(1, BUFFER_LEN);
	if (!res->buf)
		die("calloc");

	res->mr = ibv_reg_mr(res->pd, res->buf, BUFFER_LEN,
			     IBV_ACCESS_LOCAL_WRITE |
			     IBV_ACCESS_REMOTE_WRITE |
			     IBV_ACCESS_REMOTE_READ);
	if (!res->mr)
		die("ibv_reg_mr");

	/*
	 * §5.2 CQ 深度必须是 send + recv 的上限之和。
	 * 取小了会在高负载下静默丢完成事件——这是最常见的配置错误之一。
	 */
	res->cq = ibv_create_cq(id->verbs, 16, NULL, NULL, 0);
	if (!res->cq)
		die("ibv_create_cq");

	memset(&qp_attr, 0, sizeof(qp_attr));
	qp_attr.send_cq = res->cq;
	qp_attr.recv_cq = res->cq;
	qp_attr.qp_type = IBV_QPT_RC;
	qp_attr.cap.max_send_wr  = 8;
	qp_attr.cap.max_recv_wr  = 8;
	qp_attr.cap.max_send_sge = 1;
	qp_attr.cap.max_recv_sge = 1;
	/* §5.3 sq_sig_all=0 表示"默认不产生 CQE"，每笔 WR 要自己设 SIGNALED */
	qp_attr.sq_sig_all = 0;

	res->qp = ibv_create_qp(res->pd, &qp_attr);
	if (!res->qp)
		die("ibv_create_qp");

	/* ---- §6 三个 modify_qp：每个只填它那一段该填的字段 ---- */

	/* RESET -> INIT：打开"能收"之前的开关（此时仍不能收发） */
	memset(&attr, 0, sizeof(attr));
	attr.qp_state        = IBV_QPS_INIT;
	attr.pkey_index      = 0;
	attr.port_num        = id->port_num;
	attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE |
			       IBV_ACCESS_REMOTE_WRITE |
			       IBV_ACCESS_REMOTE_READ;
	rc = ibv_modify_qp(res->qp, &attr,
			   IBV_QP_STATE | IBV_QP_PKEY_INDEX |
			   IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
	check(rc, "modify_qp INIT");

	/* INIT -> RTR：填对端身份与路径参数，此时能收不能发 */
	memset(&attr, 0, sizeof(attr));
	attr.qp_state              = IBV_QPS_RTR;
	attr.path_mtu              = IBV_MTU_4096;
	attr.dest_qp_num           = id->qp->qp_num;
	attr.rq_psn                = 0;
	attr.max_dest_rd_atomic    = 8;
	attr.min_rnr_timer         = 12;
	attr.ah_attr.is_global     = 1;
	attr.ah_attr.port_num      = id->port_num;
	/*
	 * 注意：RoCE 上下面这样填是**不完整**的。
	 * is_global=1 之后还必须填 grh 的两个字段，缺一个就 EINVAL：
	 *   sgid_index  本端源 GID —— 本机端口 GID 表的「行号」，不是地址。
	 *               同一 IP 在表里有 v1/v2 两行，index 0 常是 link-local v1；
	 *               每行还绑一个 netdev，所以它也决定从哪个口发出去。
	 *               不要写死，用 ibv_query_gid_table() 按类型匹配。
	 *   dgid        对端 GID —— RoCEv2 下是 IPv4-mapped IPv6（::ffff:a.b.c.d）。
	 * 本示例走 IB 路径（有 LID），所以 GRH 只写最简形式；完整填法见
	 * 38 章正文「第三段补：GID 表和 sgid_index」。
	 */
	attr.ah_attr.grh.hop_limit = 1;
	rc = ibv_modify_qp(res->qp, &attr,
			   IBV_QP_STATE | IBV_QP_AV |
			   IBV_QP_PATH_MTU | IBV_QP_DEST_QPN |
			   IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
			   IBV_QP_MIN_RNR_TIMER);
	check(rc, "modify_qp RTR");

	/* RTR -> RTS：填重传与超时，此时才能发 */
	memset(&attr, 0, sizeof(attr));
	attr.qp_state      = IBV_QPS_RTS;
	attr.timeout       = 14;   /* 4.096us << 14 ≈ 67ms */
	attr.retry_cnt     = 7;    /* 7 = 无限重试 */
	attr.rnr_retry     = 7;    /* 7 = 无限重试，生产里通常设有限值 */
	attr.sq_psn        = 0;
	attr.max_rd_atomic = 8;
	rc = ibv_modify_qp(res->qp, &attr,
			   IBV_QP_STATE | IBV_QP_TIMEOUT |
			   IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
			   IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
	check(rc, "modify_qp RTS");

	printf("[%s] 资源就绪 qpn=%u\n",
	       id->qp ? "ok" : "?", res->qp->qp_num);
}

static void destroy_resources(struct resources *res)
{
	if (res->qp)
		ibv_destroy_qp(res->qp);
	if (res->cq)
		ibv_destroy_cq(res->cq);
	if (res->mr)
		ibv_dereg_mr(res->mr);
	if (res->pd)
		ibv_dealloc_pd(res->pd);
	free(res->buf);
}

/* ------------------------------------------------------------------ 收发 */

/*
 * §7.1 投一个 recv WR。
 * 注意 wr_id 是应用自己用的标签；这里直接用缓冲区下标。
 */
static void post_recv(struct resources *res)
{
	struct ibv_sge sge;
	struct ibv_recv_wr wr, *bad;

	memset(&sge, 0, sizeof(sge));
	sge.addr   = (uintptr_t)res->buf;
	sge.length = BUFFER_LEN;
	sge.lkey   = res->mr->lkey;

	memset(&wr, 0, sizeof(wr));
	wr.wr_id      = 0;
	wr.sg_list    = &sge;
	wr.num_sge    = 1;

	check(ibv_post_recv(res->qp, &wr, &bad), "ibv_post_recv");
}

/* §7.2 投一个 send WR。signaled=1 才会产生 CQE。 */
static void post_send(struct resources *res)
{
	struct ibv_sge sge;
	struct ibv_send_wr wr, *bad;

	memset(&sge, 0, sizeof(sge));
	sge.addr   = (uintptr_t)res->buf;
	sge.length = MSG_LEN;
	sge.lkey   = res->mr->lkey;

	memset(&wr, 0, sizeof(wr));
	wr.wr_id      = 1;
	wr.sg_list    = &sge;
	wr.num_sge    = 1;
	wr.opcode     = IBV_WR_SEND;
	wr.send_flags = IBV_SEND_SIGNALED;   /* 因为没有设 sq_sig_all */

	check(ibv_post_send(res->qp, &wr, &bad), "ibv_post_send");
}

/*
 * §7.3 轮询完成。真实程序里这里是死循环 + 超时保护。
 * 关键：必须检查 wc.status，只看 "poll 到了" 会把错误静默吞掉。
 */
static int wait_completion(struct resources *res, const char *who)
{
	struct ibv_wc wc;
	int spins = 0;

	for (;;) {
		int n = ibv_poll_cq(res->cq, 1, &wc);
		if (n < 0) {
			fprintf(stderr, "[%s] ibv_poll_cq 返回负数\n", who);
			return -1;
		}
		if (n == 0) {
			if (++spins > 100000000) {
				fprintf(stderr, "[%s] 等不到完成，放弃\n", who);
				return -1;
			}
			continue;
		}
		printf("[%s] CQE wr_id=%lu opcode=%d status=%s\n",
		       who, (unsigned long)wc.wr_id, wc.opcode,
		       wc_status_name(wc.status));
		if (wc.status != IBV_WC_SUCCESS)
			return -1;
		return 0;
	}
}

/* ------------------------------------------------------------------ 事件 */

static void on_established(struct rdma_cm_id *id, struct rdma_conn_param *param,
			   int is_server)
{
	struct resources *res = id->context;

	build_resources(id, res);

	if (is_server) {
		/* 服务器必须先投 recv WR，否则客户端的 SEND 会 RNR */
		post_recv(res);
	} else {
		snprintf(res->buf, BUFFER_LEN, "hello rdma from client pid=%d",
			 (int)getpid());
	}
	(void)param;
}

/* ------------------------------------------------------------------ 主流程 */

static int run_server(const char *port)
{
	struct rdma_event_channel *ch;
	struct rdma_cm_id *listen_id = NULL, *conn_id = NULL;
	struct rdma_cm_event *ev = NULL;
	struct sockaddr_in addr;
	struct resources res;
	int rc, done = 0;

	memset(&res, 0, sizeof(res));
	ch = rdma_create_event_channel();
	if (!ch)
		die("rdma_create_event_channel");
	check(rdma_create_id(ch, &listen_id, NULL, RDMA_PS_TCP),
	      "rdma_create_id");

	memset(&addr, 0, sizeof(addr));
	addr.sin_family      = AF_INET;
	addr.sin_port        = htons((uint16_t)atoi(port));
	addr.sin_addr.s_addr = INADDR_ANY;

	check(rdma_bind_addr(listen_id, (struct sockaddr *)&addr), "rdma_bind_addr");
	check(rdma_listen(listen_id, 1), "rdma_listen");
	printf("[server] 监听 %s\n", port);

	while (!done) {
		check(rdma_get_cm_event(ch, &ev), "rdma_get_cm_event");
		switch (ev->event) {
		case RDMA_CM_EVENT_CONNECT_REQUEST:
			conn_id = ev->id;
			conn_id->context = &res;
			printf("[server] 收到连接请求\n");
			/* 必须在 accept 之前回应，这里直接 accept */
			{
				struct rdma_conn_param cp;
				memset(&cp, 0, sizeof(cp));
				cp.responder_resources = 8;
				cp.initiator_depth     = 8;
				cp.retry_count         = 7;
				cp.rnr_retry_count     = 7;
				check(rdma_accept(conn_id, &cp), "rdma_accept");
			}
			break;
		case RDMA_CM_EVENT_ESTABLISHED:
			printf("[server] 连接建立\n");
			on_established(ev->id, &ev->param.conn, 1);
			done = 1;
			break;
		case RDMA_CM_EVENT_DISCONNECTED:
			printf("[server] 对端断开\n");
			done = 1;
			break;
		default:
			printf("[server] 事件 %d（忽略）\n", ev->event);
			break;
		}
		rdma_ack_cm_event(ev);
	}

	/* 等客户端的消息落地 */
	if (res.qp) {
		if (wait_completion(&res, "server") == 0)
			printf("[server] 收到 %d 字节：%.*s\n",
			       MSG_LEN, MSG_LEN, res.buf);
	}

	rc = res.qp ? 0 : 1;
	destroy_resources(&res);
	if (conn_id)
		rdma_destroy_id(conn_id);
	if (listen_id)
		rdma_destroy_id(listen_id);
	rdma_destroy_event_channel(ch);
	return rc;
}

static int run_client(const char *host, const char *port)
{
	struct rdma_event_channel *ch;
	struct rdma_cm_id *id = NULL;
	struct rdma_cm_event *ev = NULL;
	struct sockaddr_in addr;
	struct resources res;
	struct addrinfo hints, *ai = NULL;
	int rc, done = 0;

	memset(&res, 0, sizeof(res));

	memset(&hints, 0, sizeof(hints));
	hints.ai_family   = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	rc = getaddrinfo(host, NULL, &hints, &ai);
	if (rc)
		die("getaddrinfo");

	ch = rdma_create_event_channel();
	if (!ch)
		die("rdma_create_event_channel");
	check(rdma_create_id(ch, &id, &res, RDMA_PS_TCP), "rdma_create_id");

	memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_port   = htons((uint16_t)atoi(port));
	addr.sin_addr   = ((struct sockaddr_in *)ai->ai_addr)->sin_addr;
	freeaddrinfo(ai);

	check(rdma_resolve_addr(id, NULL, (struct sockaddr *)&addr, 2000),
	      "rdma_resolve_addr");

	while (!done) {
		check(rdma_get_cm_event(ch, &ev), "rdma_get_cm_event");
		switch (ev->event) {
		case RDMA_CM_EVENT_ADDR_RESOLVED:
			check(rdma_resolve_route(id, 2000), "rdma_resolve_route");
			break;
		case RDMA_CM_EVENT_ROUTE_RESOLVED:
			{
				struct rdma_conn_param cp;
				memset(&cp, 0, sizeof(cp));
				cp.responder_resources = 8;
				cp.initiator_depth     = 8;
				cp.retry_count         = 7;
				cp.rnr_retry_count     = 7;
				check(rdma_connect(id, &cp), "rdma_connect");
			}
			break;
		case RDMA_CM_EVENT_ESTABLISHED:
			printf("[client] 连接建立\n");
			on_established(id, &ev->param.conn, 0);
			done = 1;
			break;
		default:
			printf("[client] 事件 %d（忽略）\n", ev->event);
			break;
		}
		rdma_ack_cm_event(ev);
	}

	/* 客户端发一条消息并等自己的 CQE */
	if (res.qp) {
		post_send(&res);
		wait_completion(&res, "client");
	}

	rc = res.qp ? 0 : 1;
	destroy_resources(&res);
	if (id)
		rdma_destroy_id(id);
	rdma_destroy_event_channel(ch);
	return rc;
}

int main(int argc, char **argv)
{
	if (argc < 3) {
		fprintf(stderr,
			"用法：\n"
			"  %s server <port>\n"
			"  %s client <host> <port>\n",
			argv[0], argv[0]);
		return 2;
	}
	if (!strcmp(argv[1], "server"))
		return run_server(argv[2]);
	if (!strcmp(argv[1], "client") && argc == 4)
		return run_client(argv[2], argv[3]);

	fprintf(stderr, "参数不对\n");
	return 2;
}
