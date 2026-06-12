import request from '@/utils/request'

export function listAlarm(query) {
  return request({
    url: '/sop/alarm/list',
    method: 'get',
    params: query
  })
}

export function getAlarm(alarmId) {
  return request({
    url: '/sop/alarm/' + alarmId,
    method: 'get'
  })
}

export function addAlarm(data) {
  return request({
    url: '/sop/alarm',
    method: 'post',
    data: data
  })
}

export function updateAlarm(data) {
  return request({
    url: '/sop/alarm',
    method: 'put',
    data: data
  })
}

export function delAlarm(alarmId) {
  return request({
    url: '/sop/alarm/' + alarmId,
    method: 'delete'
  })
}
