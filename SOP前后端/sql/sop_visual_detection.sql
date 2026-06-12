-- SOP visual error-proofing detection module
-- Execute this script after the base RuoYi SQL has been initialized.

-- ----------------------------
-- 1. Product
-- ----------------------------
create table if not exists sop_product (
  product_id      bigint(20)     not null auto_increment comment 'Product ID',
  product_code    varchar(64)    not null                comment 'Product code',
  product_name    varchar(100)   not null                comment 'Product name',
  product_model   varchar(100)   default null            comment 'Product model',
  status          char(1)        default '0'             comment 'Status: 0 normal, 1 disabled',
  create_by       varchar(64)    default ''              comment 'Created by',
  create_time     datetime                              comment 'Created time',
  update_by       varchar(64)    default ''              comment 'Updated by',
  update_time     datetime                              comment 'Updated time',
  remark          varchar(500)   default null            comment 'Remark',
  primary key (product_id),
  unique key uk_sop_product_code (product_code)
) engine=innodb auto_increment=100 comment='SOP product table';

-- ----------------------------
-- 2. SOP process
-- ----------------------------
create table if not exists sop_process (
  sop_id          bigint(20)     not null auto_increment comment 'SOP ID',
  sop_code        varchar(64)    not null                comment 'SOP code',
  sop_name        varchar(100)   not null                comment 'SOP name',
  product_id      bigint(20)     not null                comment 'Product ID',
  version         varchar(32)    default 'V1.0'          comment 'SOP version',
  status          char(1)        default '0'             comment 'Status: 0 normal, 1 disabled',
  create_by       varchar(64)    default ''              comment 'Created by',
  create_time     datetime                              comment 'Created time',
  update_by       varchar(64)    default ''              comment 'Updated by',
  update_time     datetime                              comment 'Updated time',
  remark          varchar(500)   default null            comment 'Remark',
  primary key (sop_id),
  unique key uk_sop_process_code (sop_code),
  key idx_sop_process_product (product_id)
) engine=innodb auto_increment=100 comment='SOP process table';

-- ----------------------------
-- 3. SOP step
-- ----------------------------
create table if not exists sop_step (
  step_id              bigint(20)    not null auto_increment comment 'Step ID',
  sop_id               bigint(20)    not null                comment 'SOP ID',
  step_no              int(4)        not null                comment 'Step sequence number',
  step_name            varchar(100)  not null                comment 'Step name',
  expected_event       varchar(100)  not null                comment 'Expected event code',
  required_confidence  decimal(8,4)  default 0.8000          comment 'Required confidence',
  standard_duration    int(8)        default null            comment 'Standard duration in seconds',
  status               char(1)       default '0'             comment 'Status: 0 normal, 1 disabled',
  create_by            varchar(64)   default ''              comment 'Created by',
  create_time          datetime                              comment 'Created time',
  update_by            varchar(64)   default ''              comment 'Updated by',
  update_time          datetime                              comment 'Updated time',
  remark               varchar(500)  default null            comment 'Remark',
  primary key (step_id),
  unique key uk_sop_step_no (sop_id, step_no),
  key idx_sop_step_expected_event (sop_id, expected_event)
) engine=innodb auto_increment=100 comment='SOP step table';

-- ----------------------------
-- 4. Detection task
-- ----------------------------
create table if not exists sop_detection_task (
  task_id          bigint(20)    not null auto_increment comment 'Task ID',
  task_code        varchar(64)   not null                comment 'Task code',
  product_id       bigint(20)    not null                comment 'Product ID',
  sop_id           bigint(20)    not null                comment 'SOP ID',
  station_code     varchar(64)   default null            comment 'Station code',
  camera_code      varchar(64)   default null            comment 'Camera code',
  current_step_no  int(4)        default 1               comment 'Current step number',
  task_status      varchar(20)   default 'CREATED'       comment 'Task status',
  start_time       datetime                              comment 'Start time',
  end_time         datetime                              comment 'End time',
  operator_name    varchar(64)   default null            comment 'Operator name',
  create_by        varchar(64)   default ''              comment 'Created by',
  create_time      datetime                              comment 'Created time',
  update_by        varchar(64)   default ''              comment 'Updated by',
  update_time      datetime                              comment 'Updated time',
  remark           varchar(500)  default null            comment 'Remark',
  primary key (task_id),
  unique key uk_sop_task_code (task_code),
  key idx_sop_task_product (product_id),
  key idx_sop_task_sop (sop_id),
  key idx_sop_task_status (task_status)
) engine=innodb auto_increment=100 comment='SOP detection task table';

-- ----------------------------
-- 5. Task step snapshot
-- ----------------------------
create table if not exists sop_task_step (
  task_step_id         bigint(20)    not null auto_increment comment 'Task step ID',
  task_id              bigint(20)    not null                comment 'Task ID',
  step_id              bigint(20)    not null                comment 'SOP step ID',
  step_no              int(4)        not null                comment 'Step sequence number',
  step_name            varchar(100)  not null                comment 'Step name snapshot',
  expected_event       varchar(100)  not null                comment 'Expected event code snapshot',
  required_confidence  decimal(8,4)  default 0.8000          comment 'Required confidence snapshot',
  step_status          varchar(20)   default 'PENDING'       comment 'Step status',
  pass_time            datetime                              comment 'Pass time',
  event_log_id         bigint(20)    default null            comment 'Matched event log ID',
  create_by            varchar(64)   default ''              comment 'Created by',
  create_time          datetime                              comment 'Created time',
  update_by            varchar(64)   default ''              comment 'Updated by',
  update_time          datetime                              comment 'Updated time',
  remark               varchar(500)  default null            comment 'Remark',
  primary key (task_step_id),
  unique key uk_sop_task_step_no (task_id, step_no),
  key idx_sop_task_step_task (task_id),
  key idx_sop_task_step_status (task_id, step_status)
) engine=innodb auto_increment=100 comment='SOP task step snapshot table';

-- ----------------------------
-- 6. Detection event log
-- ----------------------------
create table if not exists sop_detection_event (
  event_log_id   bigint(20)    not null auto_increment comment 'Event log ID',
  request_id     varchar(64)   default null            comment 'External request ID',
  task_id        bigint(20)    default null            comment 'Task ID',
  task_code      varchar(64)   not null                comment 'Task code',
  product_code   varchar(64)   default null            comment 'Product code',
  station_code   varchar(64)   default null            comment 'Station code',
  camera_code    varchar(64)   default null            comment 'Camera code',
  event_id       varchar(64)   default null            comment 'External event ID',
  event_code     varchar(100)  not null                comment 'Event code',
  event_name     varchar(100)  default null            comment 'Event name',
  confidence     decimal(8,4)  default null            comment 'Confidence',
  event_time     datetime                              comment 'External event time',
  receive_time   datetime                              comment 'Receive time',
  image_url      varchar(500)  default null            comment 'Evidence image URL',
  raw_payload    text                                  comment 'Raw event payload',
  judge_result   varchar(30)   default null            comment 'Judge result',
  judge_message  varchar(500)  default null            comment 'Judge message',
  step_id        bigint(20)    default null            comment 'Related step ID',
  step_no        int(4)        default null            comment 'Related step number',
  create_by      varchar(64)   default ''              comment 'Created by',
  create_time    datetime                              comment 'Created time',
  update_by      varchar(64)   default ''              comment 'Updated by',
  update_time    datetime                              comment 'Updated time',
  remark         varchar(500)  default null            comment 'Remark',
  primary key (event_log_id),
  key idx_sop_event_task (task_code, event_time),
  key idx_sop_event_code (event_code),
  key idx_sop_event_result (judge_result)
) engine=innodb auto_increment=100 comment='SOP detection event log table';

-- ----------------------------
-- 7. Alarm record
-- ----------------------------
create table if not exists sop_alarm_record (
  alarm_id       bigint(20)    not null auto_increment comment 'Alarm ID',
  alarm_code     varchar(64)   not null                comment 'Alarm code',
  task_id        bigint(20)    default null            comment 'Task ID',
  task_code      varchar(64)   default null            comment 'Task code',
  product_code   varchar(64)   default null            comment 'Product code',
  station_code   varchar(64)   default null            comment 'Station code',
  camera_code    varchar(64)   default null            comment 'Camera code',
  alarm_type     varchar(30)   not null                comment 'Alarm type',
  alarm_level    varchar(20)   default 'WARN'          comment 'Alarm level',
  alarm_message  varchar(500)  not null                comment 'Alarm message',
  event_log_id   bigint(20)    default null            comment 'Event log ID',
  event_code     varchar(100)  default null            comment 'Event code',
  event_name     varchar(100)  default null            comment 'Event name',
  step_id        bigint(20)    default null            comment 'Related step ID',
  step_no        int(4)        default null            comment 'Related step number',
  alarm_time     datetime                              comment 'Alarm time',
  handle_status  varchar(20)   default 'UNHANDLED'     comment 'Handle status',
  handle_by      varchar(64)   default null            comment 'Handle by',
  handle_time    datetime                              comment 'Handle time',
  handle_remark  varchar(500)  default null            comment 'Handle remark',
  create_by      varchar(64)   default ''              comment 'Created by',
  create_time    datetime                              comment 'Created time',
  update_by      varchar(64)   default ''              comment 'Updated by',
  update_time    datetime                              comment 'Updated time',
  remark         varchar(500)  default null            comment 'Remark',
  primary key (alarm_id),
  unique key uk_sop_alarm_code (alarm_code),
  key idx_sop_alarm_task (task_code, alarm_time),
  key idx_sop_alarm_status (handle_status),
  key idx_sop_alarm_type (alarm_type)
) engine=innodb auto_increment=100 comment='SOP alarm record table';

-- ----------------------------
-- Menu and permissions
-- ----------------------------
insert into sys_menu
select 2100, 'SOP视觉检测', 0, 5, 'sop', null, '', '', 1, 0, 'M', '0', '0', '', 'eye-open', 'admin', sysdate(), '', null, 'SOP视觉检测目录'
where not exists (select 1 from sys_menu where menu_id = 2100);

insert into sys_menu
select 2101, '产品管理', 2100, 1, 'product', 'sop/product/index', '', '', 1, 0, 'C', '0', '0', 'sop:product:list', 'component', 'admin', sysdate(), '', null, '产品管理菜单'
where not exists (select 1 from sys_menu where menu_id = 2101);
insert into sys_menu
select 2102, 'SOP流程管理', 2100, 2, 'process', 'sop/process/index', '', '', 1, 0, 'C', '0', '0', 'sop:process:list', 'tree-table', 'admin', sysdate(), '', null, 'SOP流程管理菜单'
where not exists (select 1 from sys_menu where menu_id = 2102);
insert into sys_menu
select 2107, 'SOP步骤管理', 2100, 3, 'step', 'sop/step/index', '', '', 1, 0, 'C', '0', '0', 'sop:step:list', 'list', 'admin', sysdate(), '', null, 'SOP步骤管理菜单'
where not exists (select 1 from sys_menu where menu_id = 2107);
insert into sys_menu
select 2103, '检测任务管理', 2100, 4, 'task', 'sop/task/index', '', '', 1, 0, 'C', '0', '0', 'sop:task:list', 'job', 'admin', sysdate(), '', null, '检测任务管理菜单'
where not exists (select 1 from sys_menu where menu_id = 2103);
insert into sys_menu
select 2104, '实时监控工作台', 2100, 5, 'monitor', 'sop/monitor/index', '', '', 1, 0, 'C', '0', '0', 'sop:monitor:list', 'monitor', 'admin', sysdate(), '', null, '实时监控工作台菜单'
where not exists (select 1 from sys_menu where menu_id = 2104);
insert into sys_menu
select 2105, '检测事件日志', 2100, 6, 'event', 'sop/event/index', '', '', 1, 0, 'C', '0', '0', 'sop:event:list', 'log', 'admin', sysdate(), '', null, '检测事件日志菜单'
where not exists (select 1 from sys_menu where menu_id = 2105);
insert into sys_menu
select 2106, '告警记录', 2100, 7, 'alarm', 'sop/alarm/index', '', '', 1, 0, 'C', '0', '0', 'sop:alarm:list', 'bug', 'admin', sysdate(), '', null, '告警记录菜单'
where not exists (select 1 from sys_menu where menu_id = 2106);

insert into sys_menu
select 2110, '产品查询', 2101, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:product:query', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2110);
insert into sys_menu
select 2111, '产品新增', 2101, 2, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:product:add', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2111);
insert into sys_menu
select 2112, '产品修改', 2101, 3, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:product:edit', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2112);
insert into sys_menu
select 2113, '产品删除', 2101, 4, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:product:remove', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2113);
insert into sys_menu
select 2114, '产品导出', 2101, 5, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:product:export', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2114);

insert into sys_menu
select 2120, 'SOP查询', 2102, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:process:query', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2120);
insert into sys_menu
select 2121, 'SOP新增', 2102, 2, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:process:add', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2121);
insert into sys_menu
select 2122, 'SOP修改', 2102, 3, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:process:edit', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2122);
insert into sys_menu
select 2123, 'SOP删除', 2102, 4, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:process:remove', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2123);
insert into sys_menu
select 2124, 'SOP导出', 2102, 5, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:process:export', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2124);
insert into sys_menu
select 2125, '步骤查询', 2102, 6, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:step:query', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2125);
insert into sys_menu
select 21295, '步骤列表', 2102, 6, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:step:list', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 21295);
insert into sys_menu
select 2126, '步骤新增', 2102, 7, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:step:add', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2126);
insert into sys_menu
select 2127, '步骤修改', 2102, 8, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:step:edit', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2127);
insert into sys_menu
select 2128, '步骤删除', 2102, 9, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:step:remove', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2128);
insert into sys_menu
select 2129, '步骤导出', 2102, 10, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:step:export', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2129);

insert into sys_menu
select 2130, '任务查询', 2103, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:task:query', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2130);
insert into sys_menu
select 2131, '任务新增', 2103, 2, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:task:add', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2131);
insert into sys_menu
select 2132, '任务修改', 2103, 3, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:task:edit', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2132);
insert into sys_menu
select 2133, '任务删除', 2103, 4, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:task:remove', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2133);
insert into sys_menu
select 2134, '任务导出', 2103, 5, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:task:export', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2134);

insert into sys_menu
select 2140, '监控查询', 2104, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:monitor:query', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2140);

insert into sys_menu
select 2150, '事件查询', 2105, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:event:query', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2150);
insert into sys_menu
select 2151, '事件新增', 2105, 2, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:event:add', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2151);
insert into sys_menu
select 2152, '事件修改', 2105, 3, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:event:edit', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2152);
insert into sys_menu
select 2153, '事件删除', 2105, 4, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:event:remove', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2153);
insert into sys_menu
select 2154, '事件导出', 2105, 5, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:event:export', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2154);

insert into sys_menu
select 2160, '告警查询', 2106, 1, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:alarm:query', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2160);
insert into sys_menu
select 2161, '告警新增', 2106, 2, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:alarm:add', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2161);
insert into sys_menu
select 2162, '告警修改', 2106, 3, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:alarm:edit', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2162);
insert into sys_menu
select 2163, '告警删除', 2106, 4, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:alarm:remove', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2163);
insert into sys_menu
select 2164, '告警导出', 2106, 5, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:alarm:export', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2164);
insert into sys_menu
select 2165, '告警处理', 2106, 6, '#', '', '', '', 1, 0, 'F', '0', '0', 'sop:alarm:handle', '#', 'admin', sysdate(), '', null, ''
where not exists (select 1 from sys_menu where menu_id = 2165);
