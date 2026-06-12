package com.ruoyi.web.controller.sop;

import java.util.List;
import jakarta.servlet.http.HttpServletResponse;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.DeleteMapping;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.PutMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import com.ruoyi.common.annotation.Log;
import com.ruoyi.common.core.controller.BaseController;
import com.ruoyi.common.core.domain.AjaxResult;
import com.ruoyi.common.core.page.TableDataInfo;
import com.ruoyi.common.enums.BusinessType;
import com.ruoyi.common.utils.poi.ExcelUtil;
import com.ruoyi.system.domain.sop.SopProduct;
import com.ruoyi.system.service.sop.ISopProductService;

/**
 * SOP product controller
 *
 * @author ruoyi
 */
@RestController
@RequestMapping("/sop/product")
public class SopProductController extends BaseController
{
    @Autowired
    private ISopProductService sopProductService;

    @PreAuthorize("@ss.hasPermi('sop:product:list')")
    @GetMapping("/list")
    public TableDataInfo list(SopProduct sopProduct)
    {
        startPage();
        List<SopProduct> list = sopProductService.selectSopProductList(sopProduct);
        return getDataTable(list);
    }

    @Log(title = "产品管理", businessType = BusinessType.EXPORT)
    @PreAuthorize("@ss.hasPermi('sop:product:export')")
    @PostMapping("/export")
    public void export(HttpServletResponse response, SopProduct sopProduct)
    {
        List<SopProduct> list = sopProductService.selectSopProductList(sopProduct);
        ExcelUtil<SopProduct> util = new ExcelUtil<SopProduct>(SopProduct.class);
        util.exportExcel(response, list, "产品数据");
    }

    @PreAuthorize("@ss.hasPermi('sop:product:query')")
    @GetMapping(value = "/{productId}")
    public AjaxResult getInfo(@PathVariable Long productId)
    {
        return success(sopProductService.selectSopProductByProductId(productId));
    }

    @PreAuthorize("@ss.hasPermi('sop:product:add')")
    @Log(title = "产品管理", businessType = BusinessType.INSERT)
    @PostMapping
    public AjaxResult add(@Validated @RequestBody SopProduct sopProduct)
    {
        sopProduct.setCreateBy(getUsername());
        return toAjax(sopProductService.insertSopProduct(sopProduct));
    }

    @PreAuthorize("@ss.hasPermi('sop:product:edit')")
    @Log(title = "产品管理", businessType = BusinessType.UPDATE)
    @PutMapping
    public AjaxResult edit(@Validated @RequestBody SopProduct sopProduct)
    {
        sopProduct.setUpdateBy(getUsername());
        return toAjax(sopProductService.updateSopProduct(sopProduct));
    }

    @PreAuthorize("@ss.hasPermi('sop:product:remove')")
    @Log(title = "产品管理", businessType = BusinessType.DELETE)
    @DeleteMapping("/{productIds}")
    public AjaxResult remove(@PathVariable Long[] productIds)
    {
        return toAjax(sopProductService.deleteSopProductByProductIds(productIds));
    }

    @GetMapping("/optionselect")
    public AjaxResult optionselect()
    {
        return success(sopProductService.selectSopProductAll());
    }
}
