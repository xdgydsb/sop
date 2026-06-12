-- SOP detection dashboard menu patch.
-- Import this after sop_visual_detection.sql if the "SOP检测页面" menu is missing.

set names utf8mb4;

update sys_menu set order_num = 5 where menu_id = 2103;
update sys_menu set order_num = 6 where menu_id = 2104;
update sys_menu set order_num = 7 where menu_id = 2105;
update sys_menu set order_num = 8 where menu_id = 2106;

insert into sys_menu
(
    menu_id, menu_name, parent_id, order_num, path, component, query, route_name,
    is_frame, is_cache, menu_type, visible, status, perms, icon,
    create_by, create_time, update_by, update_time, remark
)
select
    2108,
    'SOP检测页面',
    2100,
    4,
    'detect',
    'sop/detect/index',
    '',
    '',
    1,
    0,
    'C',
    '0',
    '0',
    'sop:detect:view',
    'eye-open',
    'admin',
    sysdate(),
    '',
    null,
    'SOP检测页面菜单'
where not exists (select 1 from sys_menu where menu_id = 2108);

update sys_menu
set menu_name = 'SOP检测页面',
    path = 'detect',
    component = 'sop/detect/index',
    perms = 'sop:detect:view',
    icon = 'eye-open',
    visible = '0',
    status = '0'
where menu_id = 2108;
