<template>
  <div class="app-container sop-monitor">
    <el-row :gutter="12">
      <el-col :xs="24" :sm="8">
        <div class="metric-panel">
          <div class="metric-label">检测任务</div>
          <div class="metric-value">{{ taskTotal }}</div>
          <div class="metric-sub">最近任务 {{ taskList.length }} 条</div>
        </div>
      </el-col>
      <el-col :xs="24" :sm="8">
        <div class="metric-panel">
          <div class="metric-label">检测事件</div>
          <div class="metric-value">{{ eventTotal }}</div>
          <div class="metric-sub">最近事件 {{ eventList.length }} 条</div>
        </div>
      </el-col>
      <el-col :xs="24" :sm="8">
        <div class="metric-panel">
          <div class="metric-label">告警记录</div>
          <div class="metric-value danger">{{ alarmTotal }}</div>
          <div class="metric-sub">最近告警 {{ alarmList.length }} 条</div>
        </div>
      </el-col>
    </el-row>

    <el-row :gutter="12" class="mt12">
      <el-col :xs="24" :lg="12">
        <el-card shadow="never">
          <div slot="header" class="panel-title">
            <span>最近检测任务</span>
            <el-button type="text" icon="el-icon-refresh" @click="loadData">刷新</el-button>
          </div>
          <el-table v-loading="loading" :data="taskList" size="small">
            <el-table-column label="任务编码" prop="taskCode" min-width="140" show-overflow-tooltip />
            <el-table-column label="产品" prop="productName" min-width="120" show-overflow-tooltip />
            <el-table-column label="工位" prop="stationCode" width="100" />
            <el-table-column label="状态" prop="taskStatus" width="100">
              <template slot-scope="scope">
                <el-tag size="mini" :type="taskStatusType(scope.row.taskStatus)">{{ scope.row.taskStatus }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="开始时间" prop="startTime" width="160">
              <template slot-scope="scope">{{ parseTime(scope.row.startTime) }}</template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="12">
        <el-card shadow="never">
          <div slot="header" class="panel-title">
            <span>最近告警</span>
          </div>
          <el-table v-loading="loading" :data="alarmList" size="small">
            <el-table-column label="告警编码" prop="alarmCode" min-width="130" show-overflow-tooltip />
            <el-table-column label="任务编码" prop="taskCode" min-width="130" show-overflow-tooltip />
            <el-table-column label="级别" prop="alarmLevel" width="90">
              <template slot-scope="scope">
                <el-tag size="mini" :type="alarmLevelType(scope.row.alarmLevel)">{{ scope.row.alarmLevel }}</el-tag>
              </template>
            </el-table-column>
            <el-table-column label="信息" prop="alarmMessage" min-width="180" show-overflow-tooltip />
            <el-table-column label="时间" prop="alarmTime" width="160">
              <template slot-scope="scope">{{ parseTime(scope.row.alarmTime) }}</template>
            </el-table-column>
          </el-table>
        </el-card>
      </el-col>
    </el-row>

    <el-card shadow="never" class="mt12">
      <div slot="header" class="panel-title">
        <span>最近检测事件</span>
      </div>
      <el-table v-loading="loading" :data="eventList" size="small">
        <el-table-column label="请求ID" prop="requestId" min-width="150" show-overflow-tooltip />
        <el-table-column label="任务编码" prop="taskCode" min-width="140" show-overflow-tooltip />
        <el-table-column label="产品编码" prop="productCode" min-width="120" show-overflow-tooltip />
        <el-table-column label="事件编码" prop="eventCode" min-width="120" />
        <el-table-column label="事件名称" prop="eventName" min-width="140" show-overflow-tooltip />
        <el-table-column label="置信度" prop="confidence" width="90" />
        <el-table-column label="判定结果" prop="judgeResult" width="100">
          <template slot-scope="scope">
            <el-tag size="mini" :type="judgeType(scope.row.judgeResult)">{{ scope.row.judgeResult || '-' }}</el-tag>
          </template>
        </el-table-column>
        <el-table-column label="事件时间" prop="eventTime" width="160">
          <template slot-scope="scope">{{ parseTime(scope.row.eventTime) }}</template>
        </el-table-column>
      </el-table>
    </el-card>
  </div>
</template>

<script>
import { listTask } from '@/api/sop/task'
import { listEvent } from '@/api/sop/event'
import { listAlarm } from '@/api/sop/alarm'

export default {
  name: 'SopMonitor',
  data() {
    return {
      loading: false,
      taskTotal: 0,
      eventTotal: 0,
      alarmTotal: 0,
      taskList: [],
      eventList: [],
      alarmList: []
    }
  },
  created() {
    this.loadData()
  },
  methods: {
    loadData() {
      this.loading = true
      Promise.all([
        listTask({ pageNum: 1, pageSize: 8 }),
        listEvent({ pageNum: 1, pageSize: 8 }),
        listAlarm({ pageNum: 1, pageSize: 8 })
      ]).then(([taskRes, eventRes, alarmRes]) => {
        this.taskList = taskRes.rows || []
        this.eventList = eventRes.rows || []
        this.alarmList = alarmRes.rows || []
        this.taskTotal = taskRes.total || 0
        this.eventTotal = eventRes.total || 0
        this.alarmTotal = alarmRes.total || 0
        this.loading = false
      }).catch(() => {
        this.loading = false
      })
    },
    taskStatusType(status) {
      if (status === 'PASSED' || status === 'FINISHED') return 'success'
      if (status === 'FAILED') return 'danger'
      if (status === 'CREATED') return 'info'
      return ''
    },
    alarmLevelType(level) {
      if (level === 'ERROR') return 'danger'
      if (level === 'WARN') return 'warning'
      return 'info'
    },
    judgeType(result) {
      if (result === 'PASS') return 'success'
      if (result === 'FAIL') return 'danger'
      if (result === 'ALARM') return 'warning'
      return 'info'
    }
  }
}
</script>

<style scoped>
.sop-monitor .mt12 {
  margin-top: 12px;
}

.metric-panel {
  min-height: 104px;
  padding: 18px 20px;
  border: 1px solid #e6ebf5;
  background: #fff;
}

.metric-label {
  color: #606266;
  font-size: 14px;
}

.metric-value {
  margin-top: 10px;
  color: #303133;
  font-size: 30px;
  font-weight: 600;
  line-height: 1;
}

.metric-value.danger {
  color: #f56c6c;
}

.metric-sub {
  margin-top: 12px;
  color: #909399;
  font-size: 12px;
}

.panel-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
</style>
