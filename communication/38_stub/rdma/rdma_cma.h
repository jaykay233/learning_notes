/*
 * 编译检查桩头文件 —— 见 infiniband/verbs.h 顶部的说明。
 * 只声明 rdma_cm 的接口形状，不提供任何实现。
 */
#ifndef STUB_RDMA_RDMA_CMA_H
#define STUB_RDMA_RDMA_CMA_H

#include <sys/socket.h>
#include <infiniband/verbs.h>

enum rdma_port_space {
	RDMA_PS_TCP  = 0x0106,
	RDMA_PS_UDP  = 0x0111,
};

enum rdma_cm_event_type {
	RDMA_CM_EVENT_ADDR_RESOLVED = 0,
	RDMA_CM_EVENT_ADDR_ERROR = 1,
	RDMA_CM_EVENT_ROUTE_RESOLVED = 2,
	RDMA_CM_EVENT_ROUTE_ERROR = 3,
	RDMA_CM_EVENT_CONNECT_REQUEST = 4,
	RDMA_CM_EVENT_CONNECT_RESPONSE = 5,
	RDMA_CM_EVENT_CONNECT_ERROR = 6,
	RDMA_CM_EVENT_UNREACHABLE = 7,
	RDMA_CM_EVENT_REJECTED = 8,
	RDMA_CM_EVENT_ESTABLISHED = 9,
	RDMA_CM_EVENT_DISCONNECTED = 10,
	RDMA_CM_EVENT_DEVICE_REMOVAL = 11,
	RDMA_CM_EVENT_TIMEWAIT_EXIT = 12,
};

struct rdma_event_channel { int fd; };

struct rdma_cm_id {
	struct rdma_event_channel *channel;
	void                      *context;
	struct ibv_qp             *qp;
	struct ibv_context        *verbs;
	uint8_t                    port_num;
};

struct rdma_conn_param {
	uint8_t  responder_resources;
	uint8_t  initiator_depth;
	uint8_t  retry_count;
	uint8_t  rnr_retry_count;
	uint8_t  flow_control;
	uint8_t  private_data_len;
	void    *private_data;
};

union rdma_cm_event_param {
	struct rdma_conn_param conn;
};

struct rdma_cm_event {
	enum rdma_cm_event_type     event;
	int                         status;
	struct rdma_cm_id          *id;
	union rdma_cm_event_param   param;
};

struct rdma_event_channel *rdma_create_event_channel(void);
void                       rdma_destroy_event_channel(struct rdma_event_channel *ch);
int  rdma_create_id(struct rdma_event_channel *ch, struct rdma_cm_id **id,
		    void *context, enum rdma_port_space ps);
int  rdma_destroy_id(struct rdma_cm_id *id);
int  rdma_bind_addr(struct rdma_cm_id *id, struct sockaddr *addr);
int  rdma_listen(struct rdma_cm_id *id, int backlog);
int  rdma_resolve_addr(struct rdma_cm_id *id, struct sockaddr *src,
		       struct sockaddr *dst, int timeout_ms);
int  rdma_resolve_route(struct rdma_cm_id *id, int timeout_ms);
int  rdma_connect(struct rdma_cm_id *id, struct rdma_conn_param *param);
int  rdma_accept(struct rdma_cm_id *id, struct rdma_conn_param *param);
int  rdma_disconnect(struct rdma_cm_id *id);
int  rdma_get_cm_event(struct rdma_event_channel *ch, struct rdma_cm_event **ev);
int  rdma_ack_cm_event(struct rdma_cm_event *ev);

#endif /* STUB_RDMA_RDMA_CMA_H */
