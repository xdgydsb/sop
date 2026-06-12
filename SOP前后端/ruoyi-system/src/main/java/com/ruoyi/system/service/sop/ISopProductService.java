package com.ruoyi.system.service.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopProduct;

/**
 * SOP product service
 *
 * @author ruoyi
 */
public interface ISopProductService
{
    public SopProduct selectSopProductByProductId(Long productId);

    public List<SopProduct> selectSopProductList(SopProduct sopProduct);

    public List<SopProduct> selectSopProductAll();

    public int insertSopProduct(SopProduct sopProduct);

    public int updateSopProduct(SopProduct sopProduct);

    public int deleteSopProductByProductIds(Long[] productIds);

    public int deleteSopProductByProductId(Long productId);
}
