<template>
  <div class="runtime-page app-container">
    <div class="toolbar-card">
      <div class="toolbar-main">
        <el-select
          v-model="selectedProductId"
          clearable
          filterable
          class="selector-input"
          placeholder="先选择产品"
          @change="handleProductChange"
        >
          <el-option
            v-for="item in productOptions"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
        <el-select
          v-model="selectedSopId"
          clearable
          filterable
          class="selector-input"
          placeholder="再选择 SOP"
          :disabled="!selectedProductId"
        >
          <el-option
            v-for="item in filteredProcessOptions"
            :key="item.value"
            :label="item.label"
            :value="item.value"
          />
        </el-select>
        <el-button type="success" icon="el-icon-video-play" :disabled="!canStart" @click="startDetection">
          开始检测
        </el-button>
        <el-button type="danger" plain icon="el-icon-video-pause" :disabled="!canStop" @click="stopDetection">
          停止检测
        </el-button>
        <el-button type="warning" plain icon="el-icon-refresh-left" :disabled="!canReset" @click="resetDetection">
          重置检测
        </el-button>
        <el-button type="primary" plain icon="el-icon-connection" :disabled="!canBind" @click="bindCurrentRuntime">
          绑定当前实时任务
        </el-button>
        <el-button icon="el-icon-time" @click="loadLatestTaskHistory">
          读取最近历史任务
        </el-button>
      </div>

      <div class="toolbar-main mt12">
        <el-input
          v-model.trim="taskCode"
          clearable
          class="task-input"
          placeholder="也可直接输入任务编码，例如 TASK_SOP_DEMO_20260607153000"
          @keyup.enter.native="loadRuntime"
        />
        <el-button type="primary" icon="el-icon-search" @click="loadRuntime">
          加载任务
        </el-button>
        <el-switch
          v-model="autoRefresh"
          active-text="自动刷新"
          inactive-text="手动"
          @change="handleAutoRefreshChange"
        />
      </div>

      <div class="toolbar-meta">
        <span>当前产品：{{ currentProductLabel }}</span>
        <span>当前 SOP：{{ currentSopLabel }}</span>
        <span>当前任务：{{ runtime.task ? runtime.task.taskCode : '-' }}</span>
        <span>状态：{{ currentStateLabel }}</span>
        <span>模式：{{ runtime.task ? runtime.task.runtimeMode || '-' : '-' }}</span>
      </div>
    </div>

    <el-row :gutter="16" class="mt16">
      <el-col :xs="24" :lg="16">
        <el-card shadow="never" class="video-card">
          <div slot="header" class="panel-title">
            <span>实时画面</span>
            <span class="panel-side">{{ runtimeMessage }}</span>
          </div>
          <div class="video-stage">
            <img
              v-if="isMjpegStream && !streamFallbackMode"
              :src="resolvedStreamUrl"
              class="runtime-stream"
              alt="实时视频流"
              @error="handleStreamError"
              @load="handleStreamLoad"
            >
            <video
              v-else-if="streamUrl && !isMjpegStream"
              ref="streamVideo"
              :src="streamUrl"
              class="runtime-stream"
              controls
              autoplay
              muted
              playsinline
            />
            <img
              v-else-if="latestFrameUrl"
              :src="resolvedLatestFrameUrl"
              class="runtime-stream"
              alt="实时截图刷新"
              @error="handleLatestFrameError"
            >
            <div v-else class="stream-empty">
              <i class="el-icon-video-camera-solid"></i>
              <span>当前任务还没有推送实时预览流</span>
            </div>
          </div>
          <div class="frame-strip">
            <el-image
              v-if="runtime.task && runtime.task.latestFrameUrl"
              :src="runtime.task.latestFrameUrl"
              fit="cover"
              class="latest-frame"
              :preview-src-list="[runtime.task.latestFrameUrl]"
            />
            <div v-else class="frame-empty">暂无最新截图</div>
            <div class="frame-info">
              <div>FPS：{{ runtime.task && runtime.task.runtimeFps ? runtime.task.runtimeFps : '-' }}</div>
              <div>工位：{{ runtime.task ? runtime.task.stationCode || '-' : '-' }}</div>
              <div>相机：{{ runtime.task ? runtime.task.cameraCode || '-' : '-' }}</div>
            </div>
          </div>
        </el-card>

        <el-card shadow="never" class="mt16">
          <div slot="header" class="panel-title">
            <span>步骤片段</span>
            <span class="panel-side">每一步展示截图和视频片段</span>
          </div>
          <div class="step-grid">
            <div
              v-for="step in orderedSteps"
              :key="step.taskStepId || step.stepNo"
              class="step-card"
              :class="stepClass(step)"
            >
              <div class="step-top">
                <strong>S{{ step.stepNo }}</strong>
                <el-tag size="mini" :type="stepTagType(step.stepStatus)">{{ step.stepStatus || 'PENDING' }}</el-tag>
              </div>
              <div class="step-name">{{ step.stepName || ('步骤 ' + step.stepNo) }}</div>
              <div class="step-event">{{ step.expectedEvent || '-' }}</div>
              <div v-if="stepPreviewUrl(step)" class="step-thumb">
                <el-image :src="stepPreviewUrl(step)" fit="cover" :preview-src-list="[stepPreviewUrl(step)]" />
              </div>
              <div v-else class="step-thumb empty">暂无截图</div>
              <div class="step-actions">
                <el-button
                  size="mini"
                  type="primary"
                  plain
                  :disabled="!step.clipUrl"
                  @click="openClip(step.clipUrl)"
                >
                  查看片段
                </el-button>
                <span class="clip-range">{{ formatClipRange(step) }}</span>
              </div>
              <div class="step-message">{{ stepMessage(step) }}</div>
            </div>
          </div>
        </el-card>
      </el-col>

      <el-col :xs="24" :lg="8">
        <el-card shadow="never">
          <div slot="header" class="panel-title">
            <span>步骤状态</span>
            <span class="panel-side">{{ orderedSteps.length }} 步</span>
          </div>
          <div class="timeline">
            <div
              v-for="step in orderedSteps"
              :key="'line-' + (step.taskStepId || step.stepNo)"
              class="timeline-item"
            >
              <div class="timeline-index" :class="stepClass(step)">S{{ step.stepNo }}</div>
              <div class="timeline-body">
                <div class="timeline-name">{{ step.stepName || '-' }}</div>
                <div class="timeline-sub">{{ step.judgeResult || step.stepStatus || '-' }}</div>
              </div>
            </div>
          </div>
        </el-card>

        <el-card shadow="never" class="mt16">
          <div slot="header" class="panel-title">
            <span>最近事件</span>
            <span class="panel-side">{{ runtime.recentEvents.length }}</span>
          </div>
          <div class="log-list">
            <div v-for="item in runtime.recentEvents" :key="item.eventLogId" class="log-item">
              <div class="log-title">{{ item.eventName || item.eventCode }}</div>
              <div class="log-meta">{{ item.judgeResult || '-' }} | {{ formatTime(item.eventTime) }}</div>
              <div class="log-text">{{ item.judgeMessage || '-' }}</div>
            </div>
            <div v-if="!runtime.recentEvents.length" class="empty-text">暂无事件</div>
          </div>
        </el-card>

        <el-card shadow="never" class="mt16">
          <div slot="header" class="panel-title">
            <span>最近告警</span>
            <span class="panel-side">{{ runtime.recentAlarms.length }}</span>
          </div>
          <div class="log-list">
            <div v-for="item in runtime.recentAlarms" :key="item.alarmId" class="log-item alarm">
              <div class="log-title">{{ item.alarmType }}</div>
              <div class="log-meta">{{ item.alarmLevel }} | {{ formatTime(item.alarmTime) }}</div>
              <div class="log-text">{{ item.alarmMessage || '-' }}</div>
            </div>
            <div v-if="!runtime.recentAlarms.length" class="empty-text">暂无告警</div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script>
import { parseTime } from '@/utils/ruoyi'
import { listTask } from '@/api/sop/task'
import { optionselectProduct } from '@/api/sop/product'
import { optionselectProcess } from '@/api/sop/process'
import {
  getTaskRuntime,
  getCurrentRuntime,
  startRuntimeSession,
  stopRuntimeSession,
  resetRuntimeSession
} from '@/api/sop/runtime'

function emptyRuntime() {
  return {
    task: null,
    steps: [],
    recentEvents: [],
    recentAlarms: []
  }
}

export default {
  name: 'SopDetectRuntime',
  data() {
    return {
      autoRefresh: true,
      timer: null,
      loading: false,
      refreshing: false,
      streamRetryKey: 0,
      streamRetryTimer: null,
      streamWatchdogTimer: null,
      streamLoaded: false,
      streamFallbackMode: false,
      latestFrameKey: 0,
      latestFrameTimer: null,
      lastStreamUrl: '',
      productOptions: [],
      processOptions: [],
      selectedProductId: undefined,
      selectedSopId: undefined,
      taskCode: '',
      runtime: emptyRuntime()
    }
  },
  computed: {
    orderedSteps() {
      return (this.runtime.steps || []).slice().sort((a, b) => (a.stepNo || 0) - (b.stepNo || 0))
    },
    filteredProcessOptions() {
      if (!this.selectedProductId) {
        return []
      }
      return this.processOptions.filter(item => item.productId === this.selectedProductId)
    },
    streamUrl() {
      return this.runtime.task && this.runtime.task.previewStreamUrl ? this.runtime.task.previewStreamUrl : ''
    },
    latestFrameUrl() {
      return this.runtime.task && this.runtime.task.latestFrameUrl ? this.runtime.task.latestFrameUrl : ''
    },
    isMjpegStream() {
      const url = (this.streamUrl || '').toLowerCase()
      return url.endsWith('.mjpg') || url.endsWith('.mjpeg') || url.includes('/stream.mjpg') || url.includes('/video')
    },
    resolvedStreamUrl() {
      if (!this.streamUrl) {
        return ''
      }
      const joiner = this.streamUrl.includes('?') ? '&' : '?'
      return `${this.streamUrl}${joiner}_reload=${this.streamRetryKey}`
    },
    resolvedLatestFrameUrl() {
      if (!this.latestFrameUrl) {
        return ''
      }
      const joiner = this.latestFrameUrl.includes('?') ? '&' : '?'
      return `${this.latestFrameUrl}${joiner}_live=${this.latestFrameKey}`
    },
    currentProductLabel() {
      if (this.runtime.task && this.runtime.task.productName) {
        return this.runtime.task.productName
      }
      const current = this.productOptions.find(item => item.value === this.selectedProductId)
      return current ? current.label : '-'
    },
    currentSopLabel() {
      if (this.runtime.task && this.runtime.task.sopName) {
        return this.runtime.task.sopName
      }
      const current = this.processOptions.find(item => item.value === this.selectedSopId)
      return current ? current.label : '-'
    },
    currentRuntimeMode() {
      return ((this.runtime.task && this.runtime.task.runtimeMode) || '').toUpperCase()
    },
    currentTaskStatus() {
      return ((this.runtime.task && this.runtime.task.taskStatus) || '').toUpperCase()
    },
    currentStateLabel() {
      if (!this.runtime.task) return '未绑定任务'
      if (this.currentRuntimeMode === 'READY' || this.currentTaskStatus === 'CREATED') return '未开始'
      if (this.currentRuntimeMode === 'STOPPED' || this.currentTaskStatus === 'STOPPED') return '已停止'
      if (this.currentTaskStatus === 'RUNNING') return '检测中'
      if (this.currentTaskStatus === 'PASSED') return '已通过'
      if (this.currentTaskStatus === 'FAILED') return '已失败'
      if (this.currentTaskStatus === 'CANCELLED') return '已重置'
      return this.runtime.task.taskStatus || '-'
    },
    runtimeMessage() {
      if (!this.runtime.task) {
        return '等待检测端推流'
      }
      return this.runtime.task.runtimeMessage || '等待检测端推流'
    },
    canBind() {
      return !!(this.selectedProductId && this.selectedSopId)
    },
    canReset() {
      return !!(this.selectedProductId && this.selectedSopId)
    },
    canStart() {
      if (!this.selectedProductId || !this.selectedSopId) {
        return false
      }
      return !['RUNNING', 'ARMED'].includes(this.currentRuntimeMode)
    },
    canStop() {
      return !!this.runtime.task && ['RUNNING', 'ARMED'].includes(this.currentRuntimeMode)
    }
  },
  created() {
    this.selectedProductId = this.toNumberOrUndefined(this.$route.query.productId)
    this.selectedSopId = this.toNumberOrUndefined(this.$route.query.sopId)
    this.taskCode = this.$route.query.taskCode || ''
    this.initializePage()
    this.startAutoRefresh()
  },
  beforeDestroy() {
    this.stopAutoRefresh()
    this.stopStreamRetry()
    this.stopLatestFramePolling()
    this.clearStreamWatchdog()
  },
  methods: {
    async initializePage() {
      await Promise.all([this.loadProducts(), this.loadProcesses()])
      if (this.taskCode) {
        this.loadRuntime()
        return
      }
      if (this.selectedProductId && this.selectedSopId) {
        this.bindCurrentRuntime(false)
      }
    },
    loadProducts() {
      return optionselectProduct().then(response => {
        const data = response.data || []
        this.productOptions = data.map(item => ({
          label: `${item.productName}（${item.productCode}）`,
          value: item.productId
        }))
      })
    },
    loadProcesses() {
      return optionselectProcess().then(response => {
        const data = response.data || []
        this.processOptions = data.map(item => ({
          label: `${item.sopName}（${item.sopCode}）`,
          value: item.sopId,
          productId: item.productId
        }))
      })
    },
    applyRuntime(runtime) {
      const previousTaskCode = this.runtime.task ? this.runtime.task.taskCode : ''
      const previousTask = this.runtime.task || {}
      const previousMode = this.currentRuntimeMode
      const next = runtime || emptyRuntime()
      const nextTask = next.task ? { ...next.task } : null
      if (nextTask) {
        if (!nextTask.previewStreamUrl && previousTask.previewStreamUrl) {
          nextTask.previewStreamUrl = previousTask.previewStreamUrl
        }
        if (!nextTask.latestFrameUrl && previousTask.latestFrameUrl) {
          nextTask.latestFrameUrl = previousTask.latestFrameUrl
        }
      }
      this.runtime = {
        task: nextTask,
        steps: next.steps || [],
        recentEvents: next.recentEvents || [],
        recentAlarms: next.recentAlarms || []
      }
      this.taskCode = this.runtime.task ? this.runtime.task.taskCode : ''
      if (this.runtime.task) {
        this.selectedProductId = this.runtime.task.productId
        this.selectedSopId = this.runtime.task.sopId
      }
      this.syncStreamState(this.runtime.task ? this.runtime.task.previewStreamUrl || '' : '')
      const nextMode = this.currentRuntimeMode
      if ((this.taskCode && this.taskCode !== previousTaskCode) || nextMode !== previousMode) {
        this.forceStreamReconnect()
      }
    },
    buildSessionPayload() {
      return {
        productId: this.selectedProductId,
        sopId: this.selectedSopId,
        taskCode: this.runtime.task ? this.runtime.task.taskCode : '',
        stationCode: this.runtime.task && this.runtime.task.stationCode ? this.runtime.task.stationCode : 'STATION-01',
        cameraCode: this.runtime.task && this.runtime.task.cameraCode ? this.runtime.task.cameraCode : 'MV-CS050-10UC'
      }
    },
    loadRuntime() {
      if (!this.taskCode) {
        this.$message.warning('请先输入任务编码')
        return
      }
      if (this.refreshing) {
        return
      }
      this.refreshing = true
      this.loading = true
      getTaskRuntime(this.taskCode).then(response => {
        this.applyRuntime(response.data)
      }).finally(() => {
        this.loading = false
        this.refreshing = false
      })
    },
    bindCurrentRuntime(showWarning = true) {
      if (!this.selectedProductId || !this.selectedSopId) {
        if (showWarning) {
          this.$message.warning('请先选择产品和 SOP')
        }
        return
      }
      if (this.refreshing) {
        return
      }
      this.refreshing = true
      getCurrentRuntime({
        productId: this.selectedProductId,
        sopId: this.selectedSopId
      }).then(response => {
        if (!response.data || !response.data.task) {
          this.applyRuntime(emptyRuntime())
          if (showWarning) {
            this.$message.warning('当前产品/SOP 还没有实时任务，请先重置或开始检测')
          }
          return
        }
        this.applyRuntime(response.data)
      }).finally(() => {
        this.refreshing = false
      })
    },
    startDetection() {
      if (!this.canBind) {
        this.$message.warning('请先选择产品和 SOP')
        return
      }
      startRuntimeSession(this.buildSessionPayload()).then(response => {
        this.applyRuntime(response.data)
        this.forceStreamReconnect()
        this.$message.success('已进入检测状态')
      })
    },
    stopDetection() {
      if (!this.runtime.task) {
        this.$message.warning('当前没有可停止的任务')
        return
      }
      stopRuntimeSession(this.buildSessionPayload()).then(response => {
        this.applyRuntime(response.data)
        this.$message.success('检测已停止，当前保留实时预览')
      })
    },
    resetDetection() {
      if (!this.canBind) {
        this.$message.warning('请先选择产品和 SOP')
        return
      }
      resetRuntimeSession(this.buildSessionPayload()).then(response => {
        this.applyRuntime(response.data)
        this.clearVisibleStepResults()
        this.forceStreamReconnect()
        this.$message.success('已重置到未开始状态，请点击开始检测')
      })
    },
    loadLatestTaskHistory() {
      const query = { pageNum: 1, pageSize: 1 }
      if (this.selectedProductId) query.productId = this.selectedProductId
      if (this.selectedSopId) query.sopId = this.selectedSopId
      listTask(query).then(response => {
        const latest = response.rows && response.rows[0]
        if (!latest) {
          this.$message.warning('没有可加载的历史任务')
          return
        }
        this.taskCode = latest.taskCode
        this.loadRuntime()
      })
    },
    handleProductChange() {
      const currentMatched = this.filteredProcessOptions.some(item => item.value === this.selectedSopId)
      if (!currentMatched) {
        this.selectedSopId = undefined
      }
      this.applyRuntime(emptyRuntime())
    },
    handleAutoRefreshChange(value) {
      if (value) {
        this.startAutoRefresh()
      } else {
        this.stopAutoRefresh()
      }
    },
    startAutoRefresh() {
      this.stopAutoRefresh()
      this.timer = window.setInterval(() => {
        if (!this.autoRefresh) return
        if (this.taskCode) {
          this.loadRuntime()
        } else if (this.selectedProductId && this.selectedSopId) {
          this.bindCurrentRuntime(false)
        }
      }, 800)
    },
    stopAutoRefresh() {
      if (this.timer) {
        window.clearInterval(this.timer)
        this.timer = null
      }
    },
    syncStreamState(url) {
      let changed = false
      if (url !== this.lastStreamUrl) {
        this.lastStreamUrl = url
        this.streamRetryKey += 1
        this.streamFallbackMode = false
        changed = true
      }
      if (!url) {
        this.stopStreamRetry()
      }
      if (changed) {
        this.armStreamWatchdog()
      }
    },
    forceStreamReconnect() {
      this.streamFallbackMode = false
      this.streamLoaded = false
      this.streamRetryKey += 1
      this.latestFrameKey += 1
      this.stopLatestFramePolling()
      this.armStreamWatchdog()
    },
    clearVisibleStepResults() {
      if (this.runtime.task) {
        this.runtime.task = {
          ...this.runtime.task,
          currentStepNo: 1,
          taskStatus: 'CREATED',
          runtimeMode: 'READY'
        }
      }
      this.runtime.steps = (this.runtime.steps || []).map(step => ({
        ...step,
        stepStatus: 'PENDING',
        judgeResult: null,
        judgeMessage: null,
        snapshotUrl: null,
        clipUrl: null,
        clipStartMs: null,
        clipEndMs: null,
        passTime: null
      }))
    },
    handleStreamError() {
      if (!this.streamUrl) {
        return
      }
      this.streamFallbackMode = true
      this.startLatestFramePolling()
      this.startStreamRetry()
    },
    handleStreamLoad() {
      this.streamLoaded = true
      this.streamFallbackMode = false
      this.stopStreamRetry()
      this.stopLatestFramePolling()
      this.clearStreamWatchdog()
    },
    handleLatestFrameError() {
      this.latestFrameKey += 1
    },
    armStreamWatchdog() {
      this.clearStreamWatchdog()
      if (!this.streamUrl || !this.isMjpegStream || !this.latestFrameUrl) {
        return
      }
      this.streamLoaded = false
      this.streamWatchdogTimer = window.setTimeout(() => {
        if (!this.streamLoaded) {
          this.streamFallbackMode = true
          this.startLatestFramePolling()
          this.startStreamRetry()
        }
      }, 1500)
    },
    clearStreamWatchdog() {
      if (this.streamWatchdogTimer) {
        window.clearTimeout(this.streamWatchdogTimer)
        this.streamWatchdogTimer = null
      }
    },
    startStreamRetry() {
      if (!this.streamUrl || this.streamRetryTimer) {
        return
      }
      this.streamRetryTimer = window.setInterval(() => {
        this.streamFallbackMode = false
        this.streamRetryKey += 1
        this.armStreamWatchdog()
      }, 3000)
    },
    startLatestFramePolling() {
      if (!this.latestFrameUrl || this.latestFrameTimer) {
        return
      }
      this.latestFrameTimer = window.setInterval(() => {
        this.latestFrameKey += 1
      }, 200)
    },
    stopStreamRetry() {
      if (this.streamRetryTimer) {
        window.clearInterval(this.streamRetryTimer)
        this.streamRetryTimer = null
      }
    },
    stopLatestFramePolling() {
      if (this.latestFrameTimer) {
        window.clearInterval(this.latestFrameTimer)
        this.latestFrameTimer = null
      }
    },
    stepTagType(status) {
      if (status === 'PASSED' || status === 'DONE') return 'success'
      if (status === 'FAILED' || status === 'ERROR') return 'danger'
      if (status === 'RUNNING' || status === 'CURRENT') return 'warning'
      return 'info'
    },
    stepClass(step) {
      const status = (step.stepStatus || '').toUpperCase()
      if (status === 'PASSED' || status === 'DONE') return 'is-done'
      if (status === 'FAILED' || status === 'ERROR') return 'is-error'
      if (status === 'RUNNING' || status === 'CURRENT') return 'is-current'
      return 'is-pending'
    },
    stepPreviewUrl(step) {
      if (step.snapshotUrl) {
        return step.snapshotUrl
      }
      if ((step.stepStatus || '').toUpperCase() === 'RUNNING' && this.runtime.task && this.runtime.task.latestFrameUrl) {
        return this.runtime.task.latestFrameUrl
      }
      return ''
    },
    stepMessage(step) {
      if (step.judgeMessage) {
        return step.judgeMessage
      }
      if ((step.stepStatus || '').toUpperCase() === 'RUNNING') {
        return '当前步骤正在检测中'
      }
      return '等待该步骤结果'
    },
    openClip(url) {
      if (!url) return
      window.open(url, '_blank')
    },
    formatClipRange(step) {
      if (step.clipStartMs == null || step.clipEndMs == null) {
        return '未生成片段'
      }
      return `${(step.clipStartMs / 1000).toFixed(1)}s - ${(step.clipEndMs / 1000).toFixed(1)}s`
    },
    formatTime(value) {
      return value ? parseTime(value, '{y}-{m}-{d} {h}:{i}:{s}') : '-'
    },
    toNumberOrUndefined(value) {
      if (value === undefined || value === null || value === '') {
        return undefined
      }
      const result = Number(value)
      return Number.isNaN(result) ? undefined : result
    }
  }
}
</script>

<style scoped>
.runtime-page {
  background: #f5f7fb;
}

.mt12 {
  margin-top: 12px;
}

.mt16 {
  margin-top: 16px;
}

.toolbar-card {
  padding: 16px;
  border-radius: 8px;
  background: #fff;
  box-shadow: 0 1px 8px rgba(15, 35, 95, 0.08);
}

.toolbar-main {
  display: flex;
  gap: 12px;
  align-items: center;
  flex-wrap: wrap;
}

.toolbar-meta {
  display: flex;
  gap: 20px;
  margin-top: 12px;
  color: #606266;
  flex-wrap: wrap;
}

.task-input {
  width: 420px;
  max-width: 100%;
}

.selector-input {
  width: 240px;
  max-width: 100%;
}

.panel-title {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.panel-side {
  color: #909399;
  font-size: 12px;
}

.video-card {
  min-height: 400px;
}

.video-stage {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 360px;
  border-radius: 10px;
  overflow: hidden;
  background: #0f172a;
}

.runtime-stream {
  width: 100%;
  max-height: 560px;
  object-fit: contain;
  background: #000;
}

.stream-empty {
  display: flex;
  align-items: center;
  gap: 12px;
  color: #cbd5e1;
  font-size: 16px;
}

.frame-strip {
  display: flex;
  gap: 14px;
  margin-top: 16px;
  align-items: center;
}

.latest-frame {
  width: 180px;
  height: 108px;
  border-radius: 8px;
  overflow: hidden;
}

.frame-empty {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 180px;
  height: 108px;
  color: #909399;
  border: 1px dashed #dcdfe6;
  border-radius: 8px;
  background: #fafafa;
}

.frame-info {
  color: #606266;
  line-height: 1.9;
}

.step-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 14px;
}

.step-card {
  padding: 14px;
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  background: #fff;
}

.step-card.is-current {
  border-color: #e6a23c;
  box-shadow: 0 0 0 1px rgba(230, 162, 60, 0.15);
}

.step-card.is-done {
  border-color: #67c23a;
  box-shadow: 0 0 0 1px rgba(103, 194, 58, 0.15);
}

.step-card.is-error {
  border-color: #f56c6c;
  box-shadow: 0 0 0 1px rgba(245, 108, 108, 0.15);
}

.step-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.step-name {
  margin-top: 10px;
  color: #303133;
  font-weight: 600;
}

.step-event {
  margin-top: 6px;
  color: #909399;
  font-size: 12px;
}

.step-thumb {
  margin-top: 12px;
  height: 132px;
  overflow: hidden;
  border-radius: 8px;
  background: #f4f6fa;
}

.step-thumb ::v-deep img {
  width: 100%;
  height: 132px;
  object-fit: cover;
}

.step-thumb.empty {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #909399;
  border: 1px dashed #dcdfe6;
}

.step-actions {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  margin-top: 12px;
}

.clip-range {
  color: #909399;
  font-size: 12px;
}

.step-message {
  margin-top: 10px;
  min-height: 40px;
  color: #606266;
  font-size: 12px;
  line-height: 1.6;
}

.timeline-item {
  display: flex;
  gap: 12px;
  padding: 10px 0;
  border-bottom: 1px solid #f0f2f5;
}

.timeline-item:last-child {
  border-bottom: none;
}

.timeline-index {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 48px;
  height: 48px;
  border-radius: 50%;
  color: #606266;
  background: #eef2f7;
  font-weight: 700;
}

.timeline-index.is-current {
  color: #fff;
  background: #e6a23c;
}

.timeline-index.is-done {
  color: #fff;
  background: #67c23a;
}

.timeline-index.is-error {
  color: #fff;
  background: #f56c6c;
}

.timeline-body {
  flex: 1;
}

.timeline-name {
  color: #303133;
  font-weight: 600;
}

.timeline-sub {
  margin-top: 4px;
  color: #909399;
  font-size: 12px;
}

.log-list {
  max-height: 320px;
  overflow: auto;
}

.log-item {
  padding: 10px 0;
  border-bottom: 1px solid #f0f2f5;
}

.log-item:last-child {
  border-bottom: none;
}

.log-item.alarm .log-title {
  color: #f56c6c;
}

.log-title {
  color: #303133;
  font-weight: 600;
}

.log-meta,
.log-text,
.empty-text {
  margin-top: 4px;
  color: #909399;
  font-size: 12px;
}

@media (max-width: 991px) {
  .frame-strip {
    flex-direction: column;
    align-items: flex-start;
  }

  .latest-frame,
  .frame-empty {
    width: 100%;
    max-width: 320px;
  }
}
</style>
