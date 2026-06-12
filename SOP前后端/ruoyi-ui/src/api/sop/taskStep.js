import request from '@/utils/request'

export function listTaskStep(query) {
  return request({
    url: '/sop/taskStep/list',
    method: 'get',
    params: query
  })
}

export function getTaskStep(taskStepId) {
  return request({
    url: '/sop/taskStep/' + taskStepId,
    method: 'get'
  })
}

export function addTaskStep(data) {
  return request({
    url: '/sop/taskStep',
    method: 'post',
    data: data
  })
}

export function updateTaskStep(data) {
  return request({
    url: '/sop/taskStep',
    method: 'put',
    data: data
  })
}

export function delTaskStep(taskStepId) {
  return request({
    url: '/sop/taskStep/' + taskStepId,
    method: 'delete'
  })
}
