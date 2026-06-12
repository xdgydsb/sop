<template>
  <div class="app-container">
    <el-form :model="queryParams" ref="queryForm" size="small" :inline="true" v-show="showSearch" label-width="88px">
      <el-form-item v-for="field in queryFields" :key="field.prop" :label="field.label" :prop="field.prop">
        <el-select
          v-if="field.type === 'select'"
          v-model="queryParams[field.prop]"
          :placeholder="'请选择' + field.label"
          filterable
          clearable
          style="width: 220px"
        >
          <el-option
            v-for="item in field.options || []"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
        <el-input
          v-else
          v-model="queryParams[field.prop]"
          :placeholder="'请输入' + field.label"
          clearable
          style="width: 220px"
          @keyup.enter.native="handleQuery"
        />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" icon="el-icon-search" size="mini" @click="handleQuery">搜索</el-button>
        <el-button icon="el-icon-refresh" size="mini" @click="resetQuery">重置</el-button>
      </el-form-item>
    </el-form>

    <el-row :gutter="10" class="mb8">
      <el-col :span="1.5">
        <el-button
          v-if="!readonly"
          type="primary"
          plain
          icon="el-icon-plus"
          size="mini"
          @click="handleAdd"
          v-hasPermi="[permissionPrefix + ':add']"
        >新增</el-button>
      </el-col>
      <el-col :span="1.5">
        <el-button
          v-if="!readonly"
          type="success"
          plain
          icon="el-icon-edit"
          size="mini"
          :disabled="single"
          @click="handleUpdate"
          v-hasPermi="[permissionPrefix + ':edit']"
        >修改</el-button>
      </el-col>
      <el-col :span="1.5">
        <el-button
          v-if="!readonly"
          type="danger"
          plain
          icon="el-icon-delete"
          size="mini"
          :disabled="multiple"
          @click="handleDelete"
          v-hasPermi="[permissionPrefix + ':remove']"
        >删除</el-button>
      </el-col>
      <el-col :span="1.5">
        <el-button
          type="warning"
          plain
          icon="el-icon-download"
          size="mini"
          @click="handleExport"
          v-hasPermi="[permissionPrefix + ':export']"
        >导出</el-button>
      </el-col>
      <right-toolbar :showSearch.sync="showSearch" @queryTable="getList"></right-toolbar>
    </el-row>

    <el-table v-loading="loading" :data="rows" @selection-change="handleSelectionChange">
      <el-table-column type="selection" width="55" align="center" />
      <el-table-column
        v-for="column in columns"
        :key="column.prop"
        :label="column.label"
        :prop="column.prop"
        :width="column.width"
        :min-width="column.minWidth"
        :show-overflow-tooltip="column.overflow !== false"
        align="center"
      >
        <template slot-scope="scope">
          <el-tag v-if="column.tag" :type="tagType(column, scope.row[column.prop])">
            {{ formatValue(column, scope.row[column.prop]) }}
          </el-tag>
          <span v-else>{{ formatValue(column, scope.row[column.prop]) }}</span>
        </template>
      </el-table-column>
      <el-table-column v-if="!readonly" label="操作" align="center" width="150" class-name="small-padding fixed-width">
        <template slot-scope="scope">
          <el-button
            size="mini"
            type="text"
            icon="el-icon-edit"
            @click="handleUpdate(scope.row)"
            v-hasPermi="[permissionPrefix + ':edit']"
          >修改</el-button>
          <el-button
            size="mini"
            type="text"
            icon="el-icon-delete"
            @click="handleDelete(scope.row)"
            v-hasPermi="[permissionPrefix + ':remove']"
          >删除</el-button>
        </template>
      </el-table-column>
    </el-table>

    <pagination
      v-show="total > 0"
      :total="total"
      :page.sync="queryParams.pageNum"
      :limit.sync="queryParams.pageSize"
      @pagination="getList"
    />

    <el-dialog :title="title" :visible.sync="open" :width="dialogWidth" append-to-body>
      <el-form ref="form" :model="form" :rules="rules" label-width="110px">
        <el-form-item v-for="field in formFields" :key="field.prop" :label="field.label" :prop="field.prop">
          <el-select
            v-if="field.type === 'select'"
            v-model="form[field.prop]"
            :placeholder="'请选择' + field.label"
            filterable
            clearable
            style="width: 100%"
          >
            <el-option
              v-for="item in field.options || []"
              :key="item.value"
              :label="item.label"
              :value="item.value"
            />
          </el-select>
          <el-input-number
            v-else-if="field.type === 'number'"
            v-model="form[field.prop]"
            :min="field.min"
            :max="field.max"
            :precision="field.precision"
            style="width: 100%"
          />
          <el-date-picker
            v-else-if="field.type === 'datetime'"
            v-model="form[field.prop]"
            type="datetime"
            value-format="yyyy-MM-dd HH:mm:ss"
            placeholder="请选择时间"
            style="width: 100%"
          />
          <el-input
            v-else-if="field.type === 'textarea'"
            v-model="form[field.prop]"
            type="textarea"
            :rows="field.rows || 3"
            :placeholder="'请输入' + field.label"
          />
          <el-input
            v-else
            v-model="form[field.prop]"
            :placeholder="'请输入' + field.label"
          />
        </el-form-item>
        <el-form-item label="备注" prop="remark" v-if="showRemark">
          <el-input v-model="form.remark" type="textarea" placeholder="请输入备注" />
        </el-form-item>
      </el-form>
      <div slot="footer" class="dialog-footer">
        <el-button type="primary" @click="submitForm">确 定</el-button>
        <el-button @click="cancel">取 消</el-button>
      </div>
    </el-dialog>
  </div>
</template>

<script>
export default {
  name: 'SopCrud',
  props: {
    resourceName: {
      type: String,
      required: true
    },
    rowKey: {
      type: String,
      required: true
    },
    api: {
      type: Object,
      required: true
    },
    permissionPrefix: {
      type: String,
      required: true
    },
    exportUrl: {
      type: String,
      required: true
    },
    filePrefix: {
      type: String,
      required: true
    },
    queryFields: {
      type: Array,
      default: () => []
    },
    columns: {
      type: Array,
      default: () => []
    },
    formFields: {
      type: Array,
      default: () => []
    },
    defaultForm: {
      type: Object,
      default: () => ({})
    },
    dialogWidth: {
      type: String,
      default: '560px'
    },
    readonly: {
      type: Boolean,
      default: false
    },
    showRemark: {
      type: Boolean,
      default: true
    }
  },
  data() {
    return {
      loading: true,
      ids: [],
      single: true,
      multiple: true,
      showSearch: true,
      total: 0,
      rows: [],
      title: '',
      open: false,
      queryParams: {
        pageNum: 1,
        pageSize: 10
      },
      form: {},
      rules: {}
    }
  },
  created() {
    this.initQueryParams()
    this.initRules()
    this.getList()
  },
  methods: {
    initQueryParams() {
      this.queryFields.forEach(field => {
        this.$set(this.queryParams, field.prop, undefined)
      })
    },
    initRules() {
      const rules = {}
      this.formFields.forEach(field => {
        if (field.required) {
          rules[field.prop] = [{ required: true, message: field.label + '不能为空', trigger: field.type === 'select' ? 'change' : 'blur' }]
        }
      })
      this.rules = rules
    },
    getList() {
      this.loading = true
      this.api.list(this.queryParams).then(response => {
        this.rows = response.rows || []
        this.total = response.total || 0
        this.loading = false
      }).catch(() => {
        this.loading = false
      })
    },
    cancel() {
      this.open = false
      this.reset()
    },
    reset() {
      this.form = { ...this.defaultForm }
      this.resetForm('form')
    },
    handleQuery() {
      this.queryParams.pageNum = 1
      this.getList()
    },
    resetQuery() {
      this.resetForm('queryForm')
      this.handleQuery()
    },
    handleSelectionChange(selection) {
      this.ids = selection.map(item => item[this.rowKey])
      this.single = selection.length !== 1
      this.multiple = !selection.length
    },
    handleAdd() {
      this.reset()
      this.open = true
      this.title = '添加' + this.resourceName
    },
    handleUpdate(row = {}) {
      this.reset()
      const id = row[this.rowKey] || this.ids[0]
      this.api.get(id).then(response => {
        this.form = response.data || {}
        this.open = true
        this.title = '修改' + this.resourceName
      })
    },
    submitForm() {
      this.$refs['form'].validate(valid => {
        if (!valid) {
          return
        }
        const action = this.form[this.rowKey] != null ? this.api.update : this.api.add
        action(this.form).then(() => {
          this.$modal.msgSuccess(this.form[this.rowKey] != null ? '修改成功' : '新增成功')
          this.open = false
          this.getList()
        })
      })
    },
    handleDelete(row = {}) {
      const ids = row[this.rowKey] || this.ids
      this.$modal.confirm('是否确认删除' + this.resourceName + '编号为"' + ids + '"的数据项？').then(() => {
        return this.api.del(ids)
      }).then(() => {
        this.getList()
        this.$modal.msgSuccess('删除成功')
      }).catch(() => {})
    },
    handleExport() {
      this.download(this.exportUrl, {
        ...this.queryParams
      }, `${this.filePrefix}_${new Date().getTime()}.xlsx`)
    },
    formatValue(column, value) {
      if (value === null || value === undefined || value === '') {
        return '-'
      }
      if (column.type === 'datetime') {
        return this.parseTime(value)
      }
      const option = (column.options || []).find(item => item.value === value)
      return option ? option.label : value
    },
    tagType(column, value) {
      const option = (column.options || []).find(item => item.value === value)
      return option && option.tagType ? option.tagType : ''
    }
  }
}
</script>
