import request from '@/utils/request'

export function getTaskRuntime(taskCode) {
  return request({
    url: '/sop/runtime/task/' + taskCode,
    method: 'get'
  })
}

export function getCurrentRuntime(query) {
  return request({
    url: '/sop/runtime/current',
    method: 'get',
    params: query
  })
}

export function startRuntimeSession(data) {
  return request({
    url: '/sop/runtime/session/start',
    method: 'post',
    data
  })
}

export function resetRuntimeSession(data) {
  return request({
    url: '/sop/runtime/session/reset',
    method: 'post',
    data
  })
}

export function stopRuntimeSession(data) {
  return request({
    url: '/sop/runtime/session/stop',
    method: 'post',
    data
  })
}

export function syncTaskRuntime(data) {
  return request({
    url: '/sop/runtime/sync',
    method: 'post',
    data,
    headers: {
      repeatSubmit: false
    }
  })
}
