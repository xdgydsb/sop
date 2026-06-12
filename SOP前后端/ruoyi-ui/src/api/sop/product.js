import request from '@/utils/request'

export function listProduct(query) {
  return request({
    url: '/sop/product/list',
    method: 'get',
    params: query
  })
}

export function getProduct(productId) {
  return request({
    url: '/sop/product/' + productId,
    method: 'get'
  })
}

export function addProduct(data) {
  return request({
    url: '/sop/product',
    method: 'post',
    data: data
  })
}

export function updateProduct(data) {
  return request({
    url: '/sop/product',
    method: 'put',
    data: data
  })
}

export function delProduct(productId) {
  return request({
    url: '/sop/product/' + productId,
    method: 'delete'
  })
}

export function optionselectProduct() {
  return request({
    url: '/sop/product/optionselect',
    method: 'get'
  })
}
