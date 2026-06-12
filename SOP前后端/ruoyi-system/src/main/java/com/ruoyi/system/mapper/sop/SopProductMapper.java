package com.ruoyi.system.mapper.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopProduct;

/**
 * SOP product mapper
 *
 * @author ruoyi
 */
public interface SopProductMapper
{
    public SopProduct selectSopProductByProductId(Long productId);

    public List<SopProduct> selectSopProductList(SopProduct sopProduct);

    public List<SopProduct> selectSopProductAll();

    public int insertSopProduct(SopProduct sopProduct);

    public int updateSopProduct(SopProduct sopProduct);

    public int deleteSopProductByProductId(Long productId);

    public int deleteSopProductByProductIds(Long[] productIds);
}
