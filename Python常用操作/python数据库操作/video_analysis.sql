/*
 Navicat Premium Data Transfer

 Source Server         : 本地
 Source Server Type    : PostgreSQL
 Source Server Version : 140003 (140003)
 Source Host           : localhost:5432
 Source Catalog        : Industrial_Internet2
 Source Schema         : video_analysis

 Target Server Type    : PostgreSQL
 Target Server Version : 140003 (140003)
 File Encoding         : 65001

 Date: 30/05/2024 09:38:23
*/


-- ----------------------------
-- Table structure for ai_algorithm
-- ----------------------------
DROP TABLE IF EXISTS "video_analysis"."ai_algorithm";
CREATE TABLE "video_analysis"."ai_algorithm" (
  "id" varchar(64) COLLATE "pg_catalog"."default" NOT NULL,
  "algorithm_name" varchar(255) COLLATE "pg_catalog"."default",
  "algorithm_code" varchar(64) COLLATE "pg_catalog"."default",
  "algorithm_type" varchar(255) COLLATE "pg_catalog"."default",
  "algorithm_desc" varchar(255) COLLATE "pg_catalog"."default",
  "algorithm_default_config" varchar(3000) COLLATE "pg_catalog"."default",
  "deleted" int2,
  "creator" varchar(32) COLLATE "pg_catalog"."default",
  "updated_by" varchar(32) COLLATE "pg_catalog"."default",
  "create_date" timestamp(6),
  "update_date" timestamp(6)
)
;
COMMENT ON COLUMN "video_analysis"."ai_algorithm"."algorithm_name" IS '算法名称';
COMMENT ON COLUMN "video_analysis"."ai_algorithm"."algorithm_code" IS '算法编码';
COMMENT ON COLUMN "video_analysis"."ai_algorithm"."algorithm_type" IS '算法类型';
COMMENT ON COLUMN "video_analysis"."ai_algorithm"."algorithm_desc" IS '算法描述';
COMMENT ON COLUMN "video_analysis"."ai_algorithm"."algorithm_default_config" IS '算法默认配置';
COMMENT ON COLUMN "video_analysis"."ai_algorithm"."deleted" IS '逻辑删除';
COMMENT ON COLUMN "video_analysis"."ai_algorithm"."creator" IS '创建者';
COMMENT ON COLUMN "video_analysis"."ai_algorithm"."updated_by" IS '更新者';
COMMENT ON COLUMN "video_analysis"."ai_algorithm"."create_date" IS '创建时间';
COMMENT ON COLUMN "video_analysis"."ai_algorithm"."update_date" IS '更新时间';
COMMENT ON TABLE "video_analysis"."ai_algorithm" IS '算法配置表';

-- ----------------------------
-- Table structure for ai_config_push_record
-- ----------------------------
DROP TABLE IF EXISTS "video_analysis"."ai_config_push_record";
CREATE TABLE "video_analysis"."ai_config_push_record" (
  "id" varchar(64) COLLATE "pg_catalog"."default" NOT NULL,
  "node_id" varchar(64) COLLATE "pg_catalog"."default",
  "remark" varchar(255) COLLATE "pg_catalog"."default",
  "push_time" timestamp(6),
  "end_time" timestamp(6),
  "push_status" int4,
  "deleted" int2,
  "creator" varchar(32) COLLATE "pg_catalog"."default",
  "updated_by" varchar(32) COLLATE "pg_catalog"."default",
  "create_date" timestamp(6),
  "update_date" timestamp(6)
)
;
COMMENT ON COLUMN "video_analysis"."ai_config_push_record"."node_id" IS '节点ID';
COMMENT ON COLUMN "video_analysis"."ai_config_push_record"."remark" IS '备注';
COMMENT ON COLUMN "video_analysis"."ai_config_push_record"."push_time" IS '推送时间';
COMMENT ON COLUMN "video_analysis"."ai_config_push_record"."end_time" IS '推送完成时间';
COMMENT ON COLUMN "video_analysis"."ai_config_push_record"."push_status" IS '推送状态 0:推送中，1:推送成功，2:推送失败';
COMMENT ON COLUMN "video_analysis"."ai_config_push_record"."deleted" IS '逻辑删除';
COMMENT ON COLUMN "video_analysis"."ai_config_push_record"."creator" IS '创建者';
COMMENT ON COLUMN "video_analysis"."ai_config_push_record"."updated_by" IS '更新者';
COMMENT ON COLUMN "video_analysis"."ai_config_push_record"."create_date" IS '创建时间';
COMMENT ON COLUMN "video_analysis"."ai_config_push_record"."update_date" IS '更新时间';
COMMENT ON TABLE "video_analysis"."ai_config_push_record" IS '配置下发记录表';

-- ----------------------------
-- Table structure for ai_node
-- ----------------------------
DROP TABLE IF EXISTS "video_analysis"."ai_node";
CREATE TABLE "video_analysis"."ai_node" (
  "id" varchar(64) COLLATE "pg_catalog"."default" NOT NULL,
  "node_code" varchar(255) COLLATE "pg_catalog"."default",
  "node_name" varchar(255) COLLATE "pg_catalog"."default",
  "node_ip" varchar(255) COLLATE "pg_catalog"."default",
  "status" int2,
  "deleted" int2,
  "creator" varchar(32) COLLATE "pg_catalog"."default",
  "updated_by" varchar(32) COLLATE "pg_catalog"."default",
  "create_date" timestamp(6),
  "update_date" timestamp(6),
  "node_port" varchar(255) COLLATE "pg_catalog"."default",
  "last_heart_beat_time" timestamp(6),
  "error" int2,
  "error_msg" varchar(255) COLLATE "pg_catalog"."default"
)
;
COMMENT ON COLUMN "video_analysis"."ai_node"."node_code" IS '节点标识';
COMMENT ON COLUMN "video_analysis"."ai_node"."node_name" IS '节点名称';
COMMENT ON COLUMN "video_analysis"."ai_node"."node_ip" IS '节点ip';
COMMENT ON COLUMN "video_analysis"."ai_node"."status" IS '节点状态 0：离线，1：在线';
COMMENT ON COLUMN "video_analysis"."ai_node"."deleted" IS '逻辑删除';
COMMENT ON COLUMN "video_analysis"."ai_node"."creator" IS '创建者';
COMMENT ON COLUMN "video_analysis"."ai_node"."updated_by" IS '更新者';
COMMENT ON COLUMN "video_analysis"."ai_node"."create_date" IS '创建时间';
COMMENT ON COLUMN "video_analysis"."ai_node"."update_date" IS '更新时间';
COMMENT ON COLUMN "video_analysis"."ai_node"."node_port" IS '节点端口';
COMMENT ON COLUMN "video_analysis"."ai_node"."last_heart_beat_time" IS '最后一次心跳时间';
COMMENT ON COLUMN "video_analysis"."ai_node"."error" IS '故障状态 0：正常，1: 故障';
COMMENT ON COLUMN "video_analysis"."ai_node"."error_msg" IS '故障简要信息';
COMMENT ON TABLE "video_analysis"."ai_node" IS '节点管理表';

-- ----------------------------
-- Table structure for camera_binding_algorithm
-- ----------------------------
DROP TABLE IF EXISTS "video_analysis"."camera_binding_algorithm";
CREATE TABLE "video_analysis"."camera_binding_algorithm" (
  "id" varchar(64) COLLATE "pg_catalog"."default" NOT NULL,
  "resource_id" varchar(64) COLLATE "pg_catalog"."default",
  "algorithm_id" varchar(64) COLLATE "pg_catalog"."default",
  "algorithm_config" varchar(3000) COLLATE "pg_catalog"."default",
  "region_config" varchar(3000) COLLATE "pg_catalog"."default",
  "enable" bool,
  "alarm_level" varchar(255) COLLATE "pg_catalog"."default"
)
;
COMMENT ON COLUMN "video_analysis"."camera_binding_algorithm"."resource_id" IS '摄像头ID';
COMMENT ON COLUMN "video_analysis"."camera_binding_algorithm"."algorithm_id" IS '算法ID';
COMMENT ON COLUMN "video_analysis"."camera_binding_algorithm"."algorithm_config" IS '算法配置';
COMMENT ON COLUMN "video_analysis"."camera_binding_algorithm"."region_config" IS '区域配置';
COMMENT ON COLUMN "video_analysis"."camera_binding_algorithm"."enable" IS '是否启用';
COMMENT ON COLUMN "video_analysis"."camera_binding_algorithm"."alarm_level" IS '告警级别';
COMMENT ON TABLE "video_analysis"."camera_binding_algorithm" IS '绑定算法及其配置表';

-- ----------------------------
-- Table structure for camera_binding_node
-- ----------------------------
DROP TABLE IF EXISTS "video_analysis"."camera_binding_node";
CREATE TABLE "video_analysis"."camera_binding_node" (
  "id" varchar(64) COLLATE "pg_catalog"."default" NOT NULL,
  "resource_id" varchar(64) COLLATE "pg_catalog"."default",
  "node_id" varchar(64) COLLATE "pg_catalog"."default"
)
;
COMMENT ON COLUMN "video_analysis"."camera_binding_node"."resource_id" IS '摄像头ID';
COMMENT ON COLUMN "video_analysis"."camera_binding_node"."node_id" IS '节点ID';
COMMENT ON TABLE "video_analysis"."camera_binding_node" IS '摄像头和节点绑定表';

-- ----------------------------
-- Table structure for dc_dict
-- ----------------------------
DROP TABLE IF EXISTS "video_analysis"."dc_dict";
CREATE TABLE "video_analysis"."dc_dict" (
  "id" varchar(32) COLLATE "pg_catalog"."default" NOT NULL,
  "code" varchar(50) COLLATE "pg_catalog"."default",
  "en_name" varchar(50) COLLATE "pg_catalog"."default",
  "cn_name" varchar(50) COLLATE "pg_catalog"."default",
  "remark" varchar(255) COLLATE "pg_catalog"."default",
  "deleted" int2 DEFAULT 0,
  "creator_id" varchar(32) COLLATE "pg_catalog"."default",
  "create_time" timestamp(6),
  "last_update_time" timestamp(6)
)
;
COMMENT ON COLUMN "video_analysis"."dc_dict"."id" IS '主键id';
COMMENT ON COLUMN "video_analysis"."dc_dict"."code" IS '类型编码';
COMMENT ON COLUMN "video_analysis"."dc_dict"."en_name" IS '英文名称';
COMMENT ON COLUMN "video_analysis"."dc_dict"."cn_name" IS '中文名称';
COMMENT ON COLUMN "video_analysis"."dc_dict"."remark" IS '备注';
COMMENT ON COLUMN "video_analysis"."dc_dict"."deleted" IS '0:正常   1：删除';
COMMENT ON COLUMN "video_analysis"."dc_dict"."creator_id" IS '创建人id';
COMMENT ON COLUMN "video_analysis"."dc_dict"."create_time" IS '创建时间';
COMMENT ON COLUMN "video_analysis"."dc_dict"."last_update_time" IS '修改时间';
COMMENT ON TABLE "video_analysis"."dc_dict" IS '字典类型表';

-- ----------------------------
-- Table structure for dc_dict_item
-- ----------------------------
DROP TABLE IF EXISTS "video_analysis"."dc_dict_item";
CREATE TABLE "video_analysis"."dc_dict_item" (
  "id" varchar(32) COLLATE "pg_catalog"."default" NOT NULL,
  "dict_code" varchar(50) COLLATE "pg_catalog"."default",
  "cn_name" varchar(50) COLLATE "pg_catalog"."default",
  "value" varchar(16) COLLATE "pg_catalog"."default",
  "en_name" varchar(50) COLLATE "pg_catalog"."default",
  "sort" int2,
  "status" int2 DEFAULT 0,
  "defaulted" int2,
  "remark" varchar(255) COLLATE "pg_catalog"."default",
  "icon" varchar(255) COLLATE "pg_catalog"."default",
  "deleted" int2 DEFAULT 0,
  "creator_id" varchar(32) COLLATE "pg_catalog"."default",
  "create_time" timestamp(6),
  "last_update_time" timestamp(6)
)
;
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."id" IS '主键id';
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."dict_code" IS '字段编码';
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."cn_name" IS '字典项名称';
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."value" IS '字典项值';
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."en_name" IS '英文名称';
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."sort" IS '排序码';
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."status" IS '状态（0-正常 ,1-停用）';
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."defaulted" IS '是否默认（0否 1是）';
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."remark" IS '备注';
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."icon" IS '图标';
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."deleted" IS '0:正常   1：删除';
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."creator_id" IS '创建人id';
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."create_time" IS '创建时间';
COMMENT ON COLUMN "video_analysis"."dc_dict_item"."last_update_time" IS '修改时间';
COMMENT ON TABLE "video_analysis"."dc_dict_item" IS '字典明细表';

-- ----------------------------
-- Table structure for va_work_alarm_resource
-- ----------------------------
DROP TABLE IF EXISTS "video_analysis"."va_work_alarm_resource";
CREATE TABLE "video_analysis"."va_work_alarm_resource" (
  "id" varchar(64) COLLATE "pg_catalog"."default" NOT NULL,
  "work_id" varchar(255) COLLATE "pg_catalog"."default",
  "resource_id" varchar(64) COLLATE "pg_catalog"."default",
  "alarm_type" varchar(255) COLLATE "pg_catalog"."default",
  "alarm_level" varchar(255) COLLATE "pg_catalog"."default"
)
;
COMMENT ON COLUMN "video_analysis"."va_work_alarm_resource"."work_id" IS '作业ID';
COMMENT ON COLUMN "video_analysis"."va_work_alarm_resource"."resource_id" IS '报警资源ID';
COMMENT ON COLUMN "video_analysis"."va_work_alarm_resource"."alarm_type" IS '报警类型';
COMMENT ON COLUMN "video_analysis"."va_work_alarm_resource"."alarm_level" IS '报警级别';
COMMENT ON TABLE "video_analysis"."va_work_alarm_resource" IS '作业和摄像头绑定及配置表';

-- ----------------------------
-- Table structure for va_work_info
-- ----------------------------
DROP TABLE IF EXISTS "video_analysis"."va_work_info";
CREATE TABLE "video_analysis"."va_work_info" (
  "id" varchar(64) COLLATE "pg_catalog"."default" NOT NULL,
  "work_number" varchar(128) COLLATE "pg_catalog"."default",
  "work_type" varchar(64) COLLATE "pg_catalog"."default",
  "work_level" varchar(64) COLLATE "pg_catalog"."default",
  "work_org_id" varchar(64) COLLATE "pg_catalog"."default",
  "work_org_name" varchar(255) COLLATE "pg_catalog"."default",
  "work_area_id" varchar(64) COLLATE "pg_catalog"."default",
  "work_area_name" varchar(255) COLLATE "pg_catalog"."default",
  "work_status" varchar(64) COLLATE "pg_catalog"."default",
  "work_start_date_time" timestamp(6),
  "work_end_date_time" timestamp(6),
  "work_content" varchar(255) COLLATE "pg_catalog"."default",
  "work_applicant" varchar(64) COLLATE "pg_catalog"."default",
  "work_address" varchar(255) COLLATE "pg_catalog"."default",
  "coordinate" geometry(GEOMETRY),
  "work_personnel" varchar(255) COLLATE "pg_catalog"."default",
  "work_guardianship" varchar(255) COLLATE "pg_catalog"."default",
  "work_approval" varchar(255) COLLATE "pg_catalog"."default",
  "work_approval_status" varchar(64) COLLATE "pg_catalog"."default",
  "work_part" varchar(255) COLLATE "pg_catalog"."default",
  "work_describe" varchar(255) COLLATE "pg_catalog"."default",
  "jsa_analysis_results" varchar(255) COLLATE "pg_catalog"."default",
  "work_id" varchar(255) COLLATE "pg_catalog"."default"
)
;
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_number" IS '作业编号';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_type" IS '作业类型（字典）';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_level" IS '作业等级（字典）';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_org_id" IS '作业部门ID';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_org_name" IS '作业部门名称';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_area_id" IS '作业区域ID';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_area_name" IS '作业区域名字';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_status" IS '作业状态（字典）';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_start_date_time" IS '作业开始时间';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_end_date_time" IS '作业结束时间';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_content" IS '作业内容';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_applicant" IS '作业申请人';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_address" IS '作业位置';
COMMENT ON COLUMN "video_analysis"."va_work_info"."coordinate" IS '作业坐标或区域';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_personnel" IS '施工人员';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_guardianship" IS '监护人员';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_approval" IS '审批人员';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_approval_status" IS '审批状态（字典）';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_part" IS '作业部位及内容';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_describe" IS '作业描述';
COMMENT ON COLUMN "video_analysis"."va_work_info"."jsa_analysis_results" IS 'JSA分析结果';
COMMENT ON COLUMN "video_analysis"."va_work_info"."work_id" IS '作业ID';
COMMENT ON TABLE "video_analysis"."va_work_info" IS '作业信息表';

-- ----------------------------
-- Primary Key structure for table ai_algorithm
-- ----------------------------
ALTER TABLE "video_analysis"."ai_algorithm" ADD CONSTRAINT "ai_algorithm_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table ai_config_push_record
-- ----------------------------
ALTER TABLE "video_analysis"."ai_config_push_record" ADD CONSTRAINT "ai_config_push_record_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table ai_node
-- ----------------------------
ALTER TABLE "video_analysis"."ai_node" ADD CONSTRAINT "ai_node_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table camera_binding_algorithm
-- ----------------------------
ALTER TABLE "video_analysis"."camera_binding_algorithm" ADD CONSTRAINT "camera_binding_algorithm_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table camera_binding_node
-- ----------------------------
ALTER TABLE "video_analysis"."camera_binding_node" ADD CONSTRAINT "camera_binding_node_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table dc_dict
-- ----------------------------
ALTER TABLE "video_analysis"."dc_dict" ADD CONSTRAINT "sys_dict_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table dc_dict_item
-- ----------------------------
ALTER TABLE "video_analysis"."dc_dict_item" ADD CONSTRAINT "sys_dict_item_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table va_work_alarm_resource
-- ----------------------------
ALTER TABLE "video_analysis"."va_work_alarm_resource" ADD CONSTRAINT "va_work_alarm_resource_pkey" PRIMARY KEY ("id");

-- ----------------------------
-- Primary Key structure for table va_work_info
-- ----------------------------
ALTER TABLE "video_analysis"."va_work_info" ADD CONSTRAINT "va_work_info_pkey" PRIMARY KEY ("id");
