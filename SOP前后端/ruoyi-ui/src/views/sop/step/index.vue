<template>
  <sop-crud
    resource-name="SOP步骤"
    row-key="stepId"
    permission-prefix="sop:step"
    export-url="sop/step/export"
    file-prefix="step"
    :api="api"
    :query-fields="queryFields"
    :columns="columns"
    :form-fields="formFields"
    :default-form="defaultForm"
  />
</template>

<script>
import SopCrud from '../components/SopCrud'
import { listStep, getStep, addStep, updateStep, delStep } from '@/api/sop/step'
import { optionselectProcess } from '@/api/sop/process'

const statusOptions = [
  { label: '正常', value: '0', tagType: 'success' },
  { label: '停用', value: '1', tagType: 'info' }
]

export default {
  name: 'SopStep',
  components: { SopCrud },
  data() {
    return {
      processOptions: [],
      api: {
        list: listStep,
        get: getStep,
        add: addStep,
        update: updateStep,
        del: delStep
      },
      defaultForm: {
        requiredConfidence: 0.8,
        status: '0'
      }
    }
  },
  computed: {
    queryFields() {
      return [
        { label: 'SOP流程', prop: 'sopId', type: 'select', options: this.processOptions },
        { label: '步骤名称', prop: 'stepName' },
        { label: '期望事件', prop: 'expectedEvent' },
        { label: '状态', prop: 'status', type: 'select', options: statusOptions }
      ]
    },
    columns() {
      return [
        { label: '步骤ID', prop: 'stepId', width: 90 },
        { label: 'SOP名称', prop: 'sopName', minWidth: 150 },
        { label: '步骤序号', prop: 'stepNo', width: 100 },
        { label: '步骤名称', prop: 'stepName', minWidth: 160 },
        { label: '期望事件编码', prop: 'expectedEvent', minWidth: 150 },
        { label: '最低置信度', prop: 'requiredConfidence', width: 110 },
        { label: '标准时长(秒)', prop: 'standardDuration', width: 120 },
        { label: '状态', prop: 'status', width: 90, tag: true, options: statusOptions },
        { label: '创建时间', prop: 'createTime', width: 170, type: 'datetime' }
      ]
    },
    formFields() {
      return [
        { label: 'SOP流程', prop: 'sopId', type: 'select', required: true, options: this.processOptions },
        { label: '步骤序号', prop: 'stepNo', type: 'number', min: 1, required: true },
        { label: '步骤名称', prop: 'stepName', required: true },
        { label: '期望事件编码', prop: 'expectedEvent', required: true },
        { label: '最低置信度', prop: 'requiredConfidence', type: 'number', min: 0, max: 1, precision: 4, required: true },
        { label: '标准时长(秒)', prop: 'standardDuration', type: 'number', min: 0 },
        { label: '状态', prop: 'status', type: 'select', required: true, options: statusOptions }
      ]
    }
  },
  created() {
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
