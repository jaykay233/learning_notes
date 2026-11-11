/*
 * 编译检查桩头文件 —— 不是 rdma-core 的实现，也永远不要链接它。
 *
 * 用途：在没有 rdma-core 的机器（比如 macOS）上验证 38_minimal_rc.c 的
 *       语法、类型和调用形状是否自洽：
 *
 *         clang -fsyntax-only -Wall -Wextra \
 *               -I communication/38_stub communication/38_minimal_rc.c
 *
 * 它只声明接口，不提供语义。看到"编译通过"只能说明 API 用法形状对，
 * 不能说明程序在真实网卡上能跑——运行时验证见 38 章正文的边界说明。
 */
#ifndef STUB_IBV_VERBS_H
#define STUB_IBV_VERBS_H

#include <stdint.h>
#include <stddef.h>

enum ibv_wc_status {
	IBV_WC_SUCCESS = 0,
	IBV_WC_LOC_LEN_ERR = 1,
	IBV_WC_LOC_QP_OP_ERR = 2,
	IBV_WC_LOC_EEC_OP_ERR = 3,
	IBV_WC_LOC_PROT_ERR = 4,
	IBV_WC_WR_FLUSH_ERR = 5,
	IBV_WC_MW_BIND_ERR = 6,
	IBV_WC_BAD_RESP_ERR = 7,
	IBV_WC_LOC_ACCESS_ERR = 8,
	IBV_WC_REM_INV_REQ_ERR = 9,
	IBV_WC_REM_ACCESS_ERR = 10,
	IBV_WC_REM_OP_ERR = 11,
	IBV_WC_RETRY_EXC_ERR = 12,
	IBV_WC_RNR_RETRY_EXC_ERR = 13,
	IBV_WC_LOC_RDD_VIOL_ERR = 14,
	IBV_WC_REM_INV_RD_REQ_ERR = 15,
	IBV_WC_REM_ABORT_ERR = 16,
	IBV_WC_INV_EECN_ERR = 17,
	IBV_WC_INV_EEC_STATE_ERR = 18,
	IBV_WC_FATAL_ERR = 19,
	IBV_WC_RESP_TIMEOUT_ERR = 20,
	IBV_WC_GENERAL_ERR = 21,
};

enum ibv_wc_opcode {
	IBV_WC_SEND = 0,
	IBV_WC_RDMA_WRITE = 1,
	IBV_WC_RDMA_READ = 2,
	IBV_WC_RECV = 128,
	IBV_WC_RECV_RDMA_WITH_IMM = 129,
};

enum ibv_qp_type {
	IBV_QPT_RC = 2,
	IBV_QPT_UD = 3,
};

enum ibv_qp_state {
	IBV_QPS_RESET = 0,
	IBV_QPS_INIT = 1,
	IBV_QPS_RTR = 2,
	IBV_QPS_RTS = 3,
	IBV_QPS_SQD = 4,
	IBV_QPS_ERR = 7,
};

enum ibv_mtu {
	IBV_MTU_256 = 1,
	IBV_MTU_512 = 2,
	IBV_MTU_1024 = 3,
	IBV_MTU_2048 = 4,
	IBV_MTU_4096 = 5,
};

enum ibv_access_flags {
	IBV_ACCESS_LOCAL_WRITE = 1,
	IBV_ACCESS_REMOTE_WRITE = 2,
	IBV_ACCESS_REMOTE_READ = 4,
	IBV_ACCESS_REMOTE_ATOMIC = 8,
};

enum ibv_send_flags {
	IBV_SEND_FENCE = 1,
	IBV_SEND_SIGNALED = 2,
	IBV_SEND_INLINE = 4,
};

enum ibv_qp_attr_mask {
	IBV_QP_STATE = 1 << 0,
	IBV_QP_CUR_STATE = 1 << 1,
	IBV_QP_PKEY_INDEX = 1 << 2,
	IBV_QP_PORT = 1 << 3,
	IBV_QP_ACCESS_FLAGS = 1 << 4,
	IBV_QP_AV = 1 << 5,
	IBV_QP_PATH_MTU = 1 << 6,
	IBV_QP_DEST_QPN = 1 << 7,
	IBV_QP_RQ_PSN = 1 << 8,
	IBV_QP_MAX_DEST_RD_ATOMIC = 1 << 9,
	IBV_QP_MIN_RNR_TIMER = 1 << 10,
	IBV_QP_TIMEOUT = 1 << 11,
	IBV_QP_RETRY_CNT = 1 << 12,
	IBV_QP_RNR_RETRY = 1 << 13,
	IBV_QP_SQ_PSN = 1 << 14,
	IBV_QP_MAX_QP_RD_ATOMIC = 1 << 15,
};

enum ibv_wr_opcode {
	IBV_WR_RDMA_WRITE = 0,
	IBV_WR_RDMA_WRITE_WITH_IMM = 1,
	IBV_WR_SEND = 2,
	IBV_WR_SEND_WITH_IMM = 3,
	IBV_WR_RDMA_READ = 4,
	IBV_WR_ATOMIC_CMP_AND_SWP = 5,
	IBV_WR_ATOMIC_FETCH_AND_ADD = 6,
};

struct ibv_device { int dummy; };

struct ibv_context { struct ibv_device *device; };

struct ibv_pd { struct ibv_context *context; };

struct ibv_mr {
	struct ibv_pd *pd;
	void          *addr;
	size_t         length;
	uint32_t       lkey;
	uint32_t       rkey;
};

struct ibv_cq { struct ibv_context *context; };

struct ibv_qp { struct ibv_pd *pd; uint32_t qp_num; };

struct ibv_sge {
	uint64_t addr;
	uint32_t length;
	uint32_t lkey;
};

struct ibv_send_wr {
	uint64_t              wr_id;
	struct ibv_send_wr   *next;
	struct ibv_sge       *sg_list;
	int                   num_sge;
	enum ibv_wr_opcode    opcode;
	int                   send_flags;
	uint32_t              imm_data;
	uint64_t              remote_addr;
	uint32_t              rkey;
};

struct ibv_recv_wr {
	uint64_t            wr_id;
	struct ibv_recv_wr *next;
	struct ibv_sge     *sg_list;
	int                 num_sge;
};

struct ibv_wc {
	uint64_t             wr_id;
	enum ibv_wc_status   status;
	enum ibv_wc_opcode   opcode;
	uint32_t             vendor_err;
	uint32_t             byte_len;
	uint32_t             qp_num;
};

struct ibv_global_route { uint8_t hop_limit; };

struct ibv_ah_attr {
	int                     is_global;
	uint8_t                 port_num;
	struct ibv_global_route grh;
};

struct ibv_qp_cap {
	uint32_t max_send_wr;
	uint32_t max_recv_wr;
	uint32_t max_send_sge;
	uint32_t max_recv_sge;
	uint32_t max_inline_data;
};

struct ibv_qp_init_attr {
	struct ibv_cq     *send_cq;
	struct ibv_cq     *recv_cq;
	struct ibv_qp_cap  cap;
	enum ibv_qp_type   qp_type;
	int                sq_sig_all;
	uint32_t           qp_context;
};

struct ibv_qp_attr {
	enum ibv_qp_state       qp_state;
	enum ibv_qp_state       cur_qp_state;
	enum ibv_mtu            path_mtu;
	enum ibv_qp_type        qp_type;
	uint32_t                qp_access_flags;
	uint32_t                pkey_index;
	uint8_t                 port_num;
	uint32_t                dest_qp_num;
	uint32_t                rq_psn;
	uint32_t                sq_psn;
	uint32_t                max_dest_rd_atomic;
	uint32_t                max_rd_atomic;
	uint8_t                 min_rnr_timer;
	uint8_t                 timeout;
	uint8_t                 retry_cnt;
	uint8_t                 rnr_retry;
	uint8_t                 sq_draining;
	struct ibv_ah_attr      ah_attr;
};

struct ibv_pd  *ibv_alloc_pd(struct ibv_context *context);
int             ibv_dealloc_pd(struct ibv_pd *pd);
struct ibv_mr  *ibv_reg_mr(struct ibv_pd *pd, void *addr, size_t length,
			   int access);
int             ibv_dereg_mr(struct ibv_mr *mr);
struct ibv_cq  *ibv_create_cq(struct ibv_context *context, int cqe,
			      void *cq_context, void *channel, int comp_vector);
int             ibv_destroy_cq(struct ibv_cq *cq);
struct ibv_qp  *ibv_create_qp(struct ibv_pd *pd, struct ibv_qp_init_attr *attr);
int             ibv_destroy_qp(struct ibv_qp *qp);
int             ibv_modify_qp(struct ibv_qp *qp, struct ibv_qp_attr *attr,
			      int attr_mask);
int             ibv_post_send(struct ibv_qp *qp, struct ibv_send_wr *wr,
			      struct ibv_send_wr **bad_wr);
int             ibv_post_recv(struct ibv_qp *qp, struct ibv_recv_wr *wr,
			      struct ibv_recv_wr **bad_wr);
int             ibv_poll_cq(struct ibv_cq *cq, int num_entries,
			    struct ibv_wc *wc);

#endif /* STUB_IBV_VERBS_H */
