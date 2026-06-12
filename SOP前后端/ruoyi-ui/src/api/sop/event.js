import request from '@/utils/request'

export function listEvent(query) {
  return request({
    url: '/sop/event/list',
    method: 'get',
    params: query
  })
}

export function getEvent(eventLogId) {
  return request({
    url: '/sop/event/' + eventLogId,
    method: 'get'
  })
}

export function addEvent(data) {
  return request({
    url: '/sop/event',
    method: 'post',
    data: data
  })
}

export function updateEvent(data) {
  return request({
    url: '/sop/event',
    method: 'put',
    data: data
  })
}

export function delEvent(eventLogId) {
  return request({
    url: '/sop/event/' + eventLogId,
    method: 'delete'
  })
}

export function receiveEvent(data) {
  return request({
    url: '/sop/event/receive',
    method: 'post',
    data: data,
    headers: {
      repeatSubmit: false
    }
  })
}
