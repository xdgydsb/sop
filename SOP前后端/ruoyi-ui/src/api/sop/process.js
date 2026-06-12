import request from '@/utils/request'

export function listProcess(query) {
  return request({
    url: '/sop/process/list',
    method: 'get',
    params: query
  })
}

export function getProcess(sopId) {
  return request({
    url: '/sop/process/' + sopId,
    method: 'get'
  })
}

export function addProcess(data) {
  return request({
    url: '/sop/process',
    method: 'post',
    data: data
  })
}

export function updateProcess(data) {
  return request({
    url: '/sop/process',
    method: 'put',
    data: data
  })
}

export function delProcess(sopId) {
  return request({
    url: '/sop/process/' + sopId,
    method: 'delete'
  })
}

export function optionselectProcess() {
  return request({
    url: '/sop/process/optionselect',
    method: 'get'
  })
}
