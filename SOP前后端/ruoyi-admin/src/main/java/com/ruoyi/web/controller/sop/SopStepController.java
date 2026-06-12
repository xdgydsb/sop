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
import com.ruoyi.system.domain.sop.SopStep;
import com.ruoyi.system.service.sop.ISopStepService;

/**
 * SOP step controller
 *
 * @author ruoyi
 */
@RestController
@RequestMapping("/sop/step")
public class SopStepController extends BaseController
{
    @Autowired
    private ISopStepService sopStepService;

    @PreAuthorize("@ss.hasPermi('sop:step:list')")
    @GetMapping("/list")
    public TableDataInfo list(SopStep sopStep)
    {
        startPage();
        List<SopStep> list = sopStepService.selectSopStepList(sopStep);
        return getDataTable(list);
    }

    @Log(title = "SOP步骤管理", businessType = BusinessType.EXPORT)
    @PreAuthorize("@ss.hasPermi('sop:step:export')")
    @PostMapping("/export")
    public void export(HttpServletResponse response, SopStep sopStep)
    {
        List<SopStep> list = sopStepService.selectSopStepList(sopStep);
        ExcelUtil<SopStep> util = new ExcelUtil<SopStep>(SopStep.class);
        util.exportExcel(response, list, "SOP步骤数据");
    }

    @PreAuthorize("@ss.hasPermi('sop:step:query')")
    @GetMapping(value = "/{stepId}")
    public AjaxResult getInfo(@PathVariable Long stepId)
    {
        return success(sopStepService.selectSopStepByStepId(stepId));
    }

    @PreAuthorize("@ss.hasPermi('sop:step:add')")
    @Log(title = "SOP步骤管理", businessType = BusinessType.INSERT)
    @PostMapping
    public AjaxResult add(@Validated @RequestBody SopStep sopStep)
    {
        sopStep.setCreateBy(getUsername());
        return toAjax(sopStepService.insertSopStep(sopStep));
    }

    @PreAuthorize("@ss.hasPermi('sop:step:edit')")
    @Log(title = "SOP步骤管理", businessType = BusinessType.UPDATE)
    @PutMapping
    public AjaxResult edit(@Validated @RequestBody SopStep sopStep)
    {
        sopStep.setUpdateBy(getUsername());
        return toAjax(sopStepService.updateSopStep(sopStep));
    }

    @PreAuthorize("@ss.hasPermi('sop:step:remove')")
    @Log(title = "SOP步骤管理", businessType = BusinessType.DELETE)
    @DeleteMapping("/{stepIds}")
    public AjaxResult remove(@PathVariable Long[] stepIds)
    {
        return toAjax(sopStepService.deleteSopStepByStepIds(stepIds));
    }
}
