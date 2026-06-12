<template>
  <sop-crud
    resource-name="告警记录"
    row-key="alarmId"
    permission-prefix="sop:alarm"
    export-url="sop/alarm/export"
    file-prefix="alarm"
    dialog-width="680px"
    :api="api"
    :query-fields="queryFields"
    :columns="columns"
    :form-fields="formFields"
    :default-form="defaultForm"
    :show-remark="false"
  />
</template>

<script>
import SopCrud from '../components/SopCrud'
import { listAlarm, getAlarm, addAlarm, updateAlarm, delAlarm } from '@/api/sop/alarm'

const levelOptions = [
  { label: '提示', value: 'INFO', tagType: 'info' },
  { label: '警告', value: 'WARN', tagType: 'warning' },
  { label: '严重', value: 'ERROR', tagType: 'danger' }
]

const handleOptions = [
  { label: '未处理', value: 'UNHANDLED', tagType: 'danger' },
  { label: '处理中', value: 'PROCESSING', tagType: 'warning' },
  { label: '已处理', value: 'HANDLED', tagType: 'success' },
  { label: '已忽略', value: 'IGNORED', tagType: 'info' }
]

export default {
  name: 'SopAlarm',
  components: { SopCrud },
  data() {
    return {
      api: {
        list: listAlarm,
        get: getAlarm,
        add: addAlarm,
        update: updateAlarm,
        del: delAlarm
      },
      defaultForm: {
        alarmLevel: 'WARN',
        handleStatus: 'UNHANDLED'
      },
      queryFields: [
        { label: '告警编码', prop: 'alarmCode' },
        { label: '任务编码', prop: 'taskCode' },
        { label: '产品编码', prop: 'productCode' },
        { label: '处理状态', prop: 'handleStatus', type: 'select', options: handleOptions }
      ],
      columns: [
        { label: '告警ID', prop: 'alarmId', width: 90 },
        { label: '告警编码', prop: 'alarmCode', minWidth: 150 },
        { label: '任务编码', prop: 'taskCode', minWidth: 140 },
        { label: '产品编码', prop: 'productCode', minWidth: 120 },
        { label: '工位', prop: 'stationCode', minWidth: 100 },
        { label: '相机', prop: 'cameraCode', minWidth: 100 },
        { label: '类型', prop: 'alarmType', minWidth: 110 },
        { label: '级别', prop: 'alarmLevel', width: 90, tag: true, options: levelOptions },
        { label: '告警信息', prop: 'alarmMessage', minWidth: 220 },
        { label: '处理状态', prop: 'handleStatus', width: 100, tag: true, options: handleOptions },
        { label: '告警时间', prop: 'alarmTime', width: 170, type: 'datetime' }
      ],
      formFields: [
        { label: '告警编码', prop: 'alarmCode', required: true },
        { label: '任务ID', prop: 'taskId', type: 'number', min: 0 },
        { label: '任务编码', prop: 'taskCode' },
        { label: '产品编码', prop: 'productCode' },
        { label: '工位编码', prop: 'stationCode' },
        { label: '相机编码', prop: 'cameraCode' },
        { label: '告警类型', prop: 'alarmType', required: true },
        { label: '告警级别', prop: 'alarmLevel', type: 'select', required: true, options: levelOptions },
        { label: '告警信息', prop: 'alarmMessage', type: 'textarea', required: true },
        { label: '事件日志ID', prop: 'eventLogId', type: 'number', min: 0 },
        { label: '事件编码', prop: 'eventCode' },
        { label: '事件名称', prop: 'eventName' },
        { label: '步骤ID', prop: 'stepId', type: 'number', min: 0 },
        { label: '步骤序号', prop: 'stepNo', type: 'number', min: 0 },
        { label: '告警时间', prop: 'alarmTime', type: 'datetime' },
        { label: '处理状态', prop: 'handleStatus', type: 'select', required: true, options: handleOptions },
        { label: '处理人', prop: 'handleBy' },
        { label: '处理时间', prop: 'handleTime', type: 'datetime' },
        { label: '处理备注', prop: 'handleRemark', type: 'textarea' }
      ]
    }
  }
}
</script>
