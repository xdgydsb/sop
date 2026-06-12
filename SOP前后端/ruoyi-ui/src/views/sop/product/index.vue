<template>
  <sop-crud
    resource-name="产品"
    row-key="productId"
    permission-prefix="sop:product"
    export-url="sop/product/export"
    file-prefix="product"
    :api="api"
    :query-fields="queryFields"
    :columns="columns"
    :form-fields="formFields"
    :default-form="defaultForm"
  />
</template>

<script>
import SopCrud from '../components/SopCrud'
import { listProduct, getProduct, addProduct, updateProduct, delProduct } from '@/api/sop/product'

const statusOptions = [
  { label: '正常', value: '0', tagType: 'success' },
  { label: '停用', value: '1', tagType: 'info' }
]

export default {
  name: 'SopProduct',
  components: { SopCrud },
  data() {
    return {
      api: {
        list: listProduct,
        get: getProduct,
        add: addProduct,
        update: updateProduct,
        del: delProduct
      },
      defaultForm: {
        status: '0'
      },
      queryFields: [
        { label: '产品编码', prop: 'productCode' },
        { label: '产品名称', prop: 'productName' },
        { label: '状态', prop: 'status', type: 'select', options: statusOptions }
      ],
      columns: [
        { label: '产品ID', prop: 'productId', width: 90 },
        { label: '产品编码', prop: 'productCode', minWidth: 140 },
        { label: '产品名称', prop: 'productName', minWidth: 160 },
        { label: '产品型号', prop: 'productModel', minWidth: 140 },
        { label: '状态', prop: 'status', width: 90, tag: true, options: statusOptions },
        { label: '创建时间', prop: 'createTime', width: 170, type: 'datetime' },
        { label: '备注', prop: 'remark', minWidth: 160 }
      ],
      formFields: [
        { label: '产品编码', prop: 'productCode', required: true },
        { label: '产品名称', prop: 'productName', required: true },
        { label: '产品型号', prop: 'productModel' },
        { label: '状态', prop: 'status', type: 'select', required: true, options: statusOptions }
      ]
    }
  }
}
</script>
