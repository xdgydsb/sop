<template>
  <sop-crud
    resource-name="检测事件"
    row-key="eventLogId"
    permission-prefix="sop:event"
    export-url="sop/event/export"
    file-prefix="event"
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
import { listEvent, getEvent, addEvent, updateEvent, delEvent } from '@/api/sop/event'
import { optionselectProduct } from '@/api/sop/product'
import { listTask } from '@/api/sop/task'

const judgeOptions = [
  { label: '通过', value: 'PASS', tagType: 'success' },
  { label: '不通过', value: 'FAIL', tagType: 'danger' },
  { label: '已忽略', value: 'IGNORED', tagType: 'info' },
  { label: '未知', value: 'UNKNOWN', tagType: 'info' }
]

export default {
  name: 'SopEvent',
  components: { SopCrud },
  data() {
    return {
      productOptions: [],
      taskOptions: [],
      api: {
        list: listEvent,
        get: getEvent,
        add: addEvent,
        update: updateEvent,
        del: delEvent
      },
      defaultForm: {},
      columns: [
        { label: '日志ID', prop: 'eventLogId', width: 90 },
        { label: '请求ID', prop: 'requestId', minWidth: 150 },
        { label: '任务编码', prop: 'taskCode', minWidth: 150 },
        { label: '产品编码', prop: 'productCode', minWidth: 120 },
        { label: '工位', prop: 'stationCode', minWidth: 110 },
        { label: '相机', prop: 'cameraCode', minWidth: 110 },
        { label: '事件编码', prop: 'eventCode', minWidth: 130 },
        { label: '事件名称', prop: 'eventName', minWidth: 140 },
        { label: '置信度', prop: 'confidence', width: 90 },
        { label: '判定结果', prop: 'judgeResult', width: 100, tag: true, options: judgeOptions },
        { label: '步骤', prop: 'stepNo', width: 80 },
        { label: '事件时间', prop: 'eventTime', width: 170, type: 'datetime' }
      ],
      formFields: [
        { label: '请求ID', prop: 'requestId' },
        { label: '任务ID', prop: 'taskId', type: 'number', min: 0 },
        { label: '任务编码', prop: 'taskCode' },
        { label: '产品编码', prop: 'productCode' },
        { label: '工位编码', prop: 'stationCode' },
        { label: '相机编码', prop: 'cameraCode' },
        { label: '事件ID', prop: 'eventId' },
        { label: '事件编码', prop: 'eventCode', required: true },
        { label: '事件名称', prop: 'eventName', required: true },
        { label: '置信度', prop: 'confidence', type: 'number', min: 0, max: 1, precision: 4 },
        { label: '事件时间', prop: 'eventTime', type: 'datetime' },
        { label: '接收时间', prop: 'receiveTime', type: 'datetime' },
        { label: '图片地址', prop: 'imageUrl' },
        { label: '判定结果', prop: 'judgeResult', type: 'select', options: judgeOptions },
        { label: '判定信息', prop: 'judgeMessage', type: 'textarea' },
        { label: '步骤ID', prop: 'stepId', type: 'number', min: 0 },
        { label: '步骤序号', prop: 'stepNo', type: 'number', min: 0 },
        { label: '原始报文', prop: 'rawPayload', type: 'textarea', rows: 4 }
      ]
    }
  },
  computed: {
    queryFields() {
      return [
        { label: '任务编码', prop: 'taskCode', type: 'select', options: this.taskOptions },
        { label: '产品', prop: 'productCode', type: 'select', options: this.productOptions },
        { label: '事件编码', prop: 'eventCode' },
        { label: '判定结果', prop: 'judgeResult', type: 'select', options: judgeOptions }
      ]
    }
  },
  created() {
    this.loadProductOptions()
    this.loadTaskOptions()
  },
  methods: {
    loadProductOptions() {
      optionselectProduct().then(response => {
        const data = response.data || []
        this.productOptions = data.map(item => ({
          label: item.productName + '（' + item.productCode + '）',
          value: item.productCode
        }))
      })
    },
    loadTaskOptions() {
      listTask({ pageNum: 1, pageSize: 1000 }).then(response => {
        const data = response.rows || []
        this.taskOptions = data.map(item => ({
          label: item.taskCode + (item.productName ? '（' + item.productName + '）' : ''),
          value: item.taskCode
        }))
      })
    }
  }
}
</script>
