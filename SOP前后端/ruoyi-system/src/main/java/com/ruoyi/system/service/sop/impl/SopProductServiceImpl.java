package com.ruoyi.system.service.sop.impl;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.ruoyi.system.domain.sop.SopProduct;
import com.ruoyi.system.mapper.sop.SopProductMapper;
import com.ruoyi.system.service.sop.ISopProductService;

/**
 * SOP product service implementation
 *
 * @author ruoyi
 */
@Service
public class SopProductServiceImpl implements ISopProductService
{
    @Autowired
    private SopProductMapper sopProductMapper;

    @Override
    public SopProduct selectSopProductByProductId(Long productId)
    {
        return sopProductMapper.selectSopProductByProductId(productId);
    }

    @Override
    public List<SopProduct> selectSopProductList(SopProduct sopProduct)
    {
        return sopProductMapper.selectSopProductList(sopProduct);
    }

    @Override
    public List<SopProduct> selectSopProductAll()
    {
        return sopProductMapper.selectSopProductAll();
    }

    @Override
    public int insertSopProduct(SopProduct sopProduct)
    {
        return sopProductMapper.insertSopProduct(sopProduct);
    }

    @Override
    public int updateSopProduct(SopProduct sopProduct)
    {
        return sopProductMapper.updateSopProduct(sopProduct);
    }

    @Override
    public int deleteSopProductByProductIds(Long[] productIds)
    {
        return sopProductMapper.deleteSopProductByProductIds(productIds);
    }

    @Override
    public int deleteSopProductByProductId(Long productId)
    {
        return sopProductMapper.deleteSopProductByProductId(productId);
    }
}
