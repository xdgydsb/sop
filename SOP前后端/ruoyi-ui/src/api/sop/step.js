import request from '@/utils/request'

export function listStep(query) {
  return request({
    url: '/sop/step/list',
    method: 'get',
    params: query
  })
}

export function getStep(stepId) {
  return request({
    url: '/sop/step/' + stepId,
    method: 'get'
  })
}

export function addStep(data) {
  return request({
    url: '/sop/step',
    method: 'post',
    data: data
  })
}

export function updateStep(data) {
  return request({
    url: '/sop/step',
    method: 'put',
    data: data
  })
}

export function delStep(stepId) {
  return request({
    url: '/sop/step/' + stepId,
    method: 'delete'
  })
}
