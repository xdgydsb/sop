-- Runtime integration upgrade for SOP visual detection.
-- Run this after sop_visual_detection.sql.

alter table sop_detection_task
  add column if not exists preview_stream_url varchar(500) default null comment 'Runtime preview stream URL' after operator_name;

alter table sop_detection_task
  add column if not exists latest_frame_url varchar(500) default null comment 'Latest frame URL' after preview_stream_url;

alter table sop_detection_task
  add column if not exists runtime_mode varchar(32) default null comment 'Runtime mode' after latest_frame_url;

alter table sop_detection_task
  add column if not exists runtime_message varchar(500) default null comment 'Runtime status message' after runtime_mode;

alter table sop_detection_task
  add column if not exists runtime_fps decimal(10,2) default null comment 'Runtime FPS' after runtime_message;

alter table sop_task_step
  add column if not exists snapshot_url varchar(500) default null comment 'Latest snapshot URL' after event_log_id;

alter table sop_task_step
  add column if not exists clip_url varchar(500) default null comment 'Step clip URL' after snapshot_url;

alter table sop_task_step
  add column if not exists clip_start_ms bigint(20) default null comment 'Clip start offset in milliseconds' after clip_url;

alter table sop_task_step
  add column if not exists clip_end_ms bigint(20) default null comment 'Clip end offset in milliseconds' after clip_start_ms;

alter table sop_task_step
  add column if not exists judge_result varchar(30) default null comment 'Latest judge result' after clip_end_ms;

alter table sop_task_step
  add column if not exists judge_message varchar(500) default null comment 'Latest judge message' after judge_result;
