<template>
  <sop-crud
    resource-name="检测任务"
    row-key="taskId"
    permission-prefix="sop:task"
    export-url="sop/task/export"
    file-prefix="task"
    dialog-width="640px"
    :api="api"
    :query-fields="queryFields"
    :columns="columns"
    :form-fields="formFields"
    :default-form="defaultForm"
  />
</template>

<script>
import SopCrud from '../components/SopCrud'
import { listTask, getTask, addTask, updateTask, delTask } from '@/api/sop/task'
import { optionselectProduct } from '@/api/sop/product'
import { optionselectProcess } from '@/api/sop/process'

const taskStatusOptions = [
  { label: '已创建', value: 'CREATED', tagType: 'info' },
  { label: '检测中', value: 'RUNNING', tagType: '' },
  { label: '已通过', value: 'PASSED', tagType: 'success' },
  { label: '异常', value: 'FAILED', tagType: 'danger' },
  { label: '已结束', value: 'FINISHED', tagType: 'success' }
]

export default {
  name: 'SopTask',
  components: { SopCrud },
  data() {
    return {
      productOptions: [],
      processOptions: [],
      api: {
        list: listTask,
        get: getTask,
        add: addTask,
        update: updateTask,
        del: delTask
      },
      defaultForm: {
        taskStatus: 'CREATED'
      }
    }
  },
  computed: {
    queryFields() {
      return [
        { label: '任务编码', prop: 'taskCode' },
        { label: '产品编码', prop: 'productCode' },
        { label: '工位编码', prop: 'stationCode' },
        { label: '任务状态', prop: 'taskStatus', type: 'select', options: taskStatusOptions }
      ]
    },
    columns() {
      return [
        { label: '任务ID', prop: 'taskId', width: 90 },
        { label: '任务编码', prop: 'taskCode', minWidth: 150 },
        { label: '产品编码', prop: 'productCode', minWidth: 120 },
        { label: '产品名称', prop: 'productName', minWidth: 140 },
        { label: 'SOP名称', prop: 'sopName', minWidth: 150 },
        { label: '工位', prop: 'stationCode', minWidth: 110 },
        { label: '相机', prop: 'cameraCode', minWidth: 110 },
        { label: '当前步骤', prop: 'currentStepNo', width: 90 },
        { label: '状态', prop: 'taskStatus', width: 100, tag: true, options: taskStatusOptions },
        { label: '开始时间', prop: 'startTime', width: 170, type: 'datetime' },
        { label: '结束时间', prop: 'endTime', width: 170, type: 'datetime' }
      ]
    },
    formFields() {
      return [
        { label: '任务编码', prop: 'taskCode', required: true },
        { label: '所属产品', prop: 'productId', type: 'select', required: true, options: this.productOptions },
        { label: 'SOP流程', prop: 'sopId', type: 'select', required: true, options: this.processOptions },
        { label: '工位编码', prop: 'stationCode', required: true },
        { label: '相机编码', prop: 'cameraCode', required: true },
        { label: '当前步骤', prop: 'currentStepNo', type: 'number', min: 0 },
        { label: '任务状态', prop: 'taskStatus', type: 'select', required: true, options: taskStatusOptions },
        { label: '开始时间', prop: 'startTime', type: 'datetime' },
        { label: '结束时间', prop: 'endTime', type: 'datetime' },
        { label: '操作员', prop: 'operatorName' }
      ]
    }
  },
  created() {
    optionselectProduct().then(response => {
      const data = response.data || []
      this.productOptions = data.map(item => ({
        label: item.productName + '（' + item.productCode + '）',
        value: item.productId
      }))
    })
    optionselectProcess().then(response => {
      const data = response.data || []
      this.processOptions = data.map(item => ({
        label: item.sopName + '（' + item.sopCode + '）',
        value: item.sopId
      }))
    })
  }
}
</script>
