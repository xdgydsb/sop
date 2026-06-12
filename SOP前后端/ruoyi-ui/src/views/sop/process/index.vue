<template>
  <sop-crud
    resource-name="SOP流程"
    row-key="sopId"
    permission-prefix="sop:process"
    export-url="sop/process/export"
    file-prefix="process"
    :api="api"
    :query-fields="queryFields"
    :columns="columns"
    :form-fields="formFields"
    :default-form="defaultForm"
  />
</template>

<script>
import SopCrud from '../components/SopCrud'
import { listProcess, getProcess, addProcess, updateProcess, delProcess } from '@/api/sop/process'
import { optionselectProduct } from '@/api/sop/product'

const statusOptions = [
  { label: '正常', value: '0', tagType: 'success' },
  { label: '停用', value: '1', tagType: 'info' }
]

export default {
  name: 'SopProcess',
  components: { SopCrud },
  data() {
    return {
      productOptions: [],
      api: {
        list: listProcess,
        get: getProcess,
        add: addProcess,
        update: updateProcess,
        del: delProcess
      },
      defaultForm: {
        version: 'V1.0',
        status: '0'
      }
    }
  },
  computed: {
    queryFields() {
      return [
        { label: 'SOP编码', prop: 'sopCode' },
        { label: 'SOP名称', prop: 'sopName' },
        { label: '产品', prop: 'productId', type: 'select', options: this.productOptions },
        { label: '状态', prop: 'status', type: 'select', options: statusOptions }
      ]
    },
    columns() {
      return [
        { label: 'SOP ID', prop: 'sopId', width: 90 },
        { label: 'SOP编码', prop: 'sopCode', minWidth: 140 },
        { label: 'SOP名称', prop: 'sopName', minWidth: 160 },
        { label: '产品编码', prop: 'productCode', minWidth: 130 },
        { label: '产品名称', prop: 'productName', minWidth: 150 },
        { label: '版本', prop: 'version', width: 100 },
        { label: '状态', prop: 'status', width: 90, tag: true, options: statusOptions },
        { label: '创建时间', prop: 'createTime', width: 170, type: 'datetime' }
      ]
    },
    formFields() {
      return [
        { label: 'SOP编码', prop: 'sopCode', required: true },
        { label: 'SOP名称', prop: 'sopName', required: true },
        { label: '所属产品', prop: 'productId', type: 'select', required: true, options: this.productOptions },
        { label: '版本', prop: 'version', required: true },
        { label: '状态', prop: 'status', type: 'select', required: true, options: statusOptions }
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
  }
}
</script>
