import request from '@/utils/request'

export function listTask(query) {
  return request({
    url: '/sop/task/list',
    method: 'get',
    params: query
  })
}

export function getTask(taskId) {
  return request({
    url: '/sop/task/' + taskId,
    method: 'get'
  })
}

export function addTask(data) {
  return request({
    url: '/sop/task',
    method: 'post',
    data: data
  })
}

export function updateTask(data) {
  return request({
    url: '/sop/task',
    method: 'put',
    data: data
  })
}

export function delTask(taskId) {
  return request({
    url: '/sop/task/' + taskId,
    method: 'delete'
  })
}
