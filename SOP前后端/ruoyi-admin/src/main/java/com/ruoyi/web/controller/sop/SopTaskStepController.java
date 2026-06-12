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
import com.ruoyi.system.domain.sop.SopTaskStep;
import com.ruoyi.system.service.sop.ISopTaskStepService;

/**
 * SOP task step snapshot controller
 *
 * @author ruoyi
 */
@RestController
@RequestMapping("/sop/taskStep")
public class SopTaskStepController extends BaseController
{
    @Autowired
    private ISopTaskStepService sopTaskStepService;

    @PreAuthorize("@ss.hasPermi('sop:task:query')")
    @GetMapping("/list")
    public TableDataInfo list(SopTaskStep sopTaskStep)
    {
        startPage();
        List<SopTaskStep> list = sopTaskStepService.selectSopTaskStepList(sopTaskStep);
        return getDataTable(list);
    }

    @Log(title = "任务步骤快照", businessType = BusinessType.EXPORT)
    @PreAuthorize("@ss.hasPermi('sop:task:export')")
    @PostMapping("/export")
    public void export(HttpServletResponse response, SopTaskStep sopTaskStep)
    {
        List<SopTaskStep> list = sopTaskStepService.selectSopTaskStepList(sopTaskStep);
        ExcelUtil<SopTaskStep> util = new ExcelUtil<SopTaskStep>(SopTaskStep.class);
        util.exportExcel(response, list, "任务步骤快照数据");
    }

    @PreAuthorize("@ss.hasPermi('sop:task:query')")
    @GetMapping(value = "/{taskStepId}")
    public AjaxResult getInfo(@PathVariable Long taskStepId)
    {
        return success(sopTaskStepService.selectSopTaskStepByTaskStepId(taskStepId));
    }

    @PreAuthorize("@ss.hasPermi('sop:task:add')")
    @Log(title = "任务步骤快照", businessType = BusinessType.INSERT)
    @PostMapping
    public AjaxResult add(@Validated @RequestBody SopTaskStep sopTaskStep)
    {
        sopTaskStep.setCreateBy(getUsername());
        return toAjax(sopTaskStepService.insertSopTaskStep(sopTaskStep));
    }

    @PreAuthorize("@ss.hasPermi('sop:task:edit')")
    @Log(title = "任务步骤快照", businessType = BusinessType.UPDATE)
    @PutMapping
    public AjaxResult edit(@Validated @RequestBody SopTaskStep sopTaskStep)
    {
        sopTaskStep.setUpdateBy(getUsername());
        return toAjax(sopTaskStepService.updateSopTaskStep(sopTaskStep));
    }

    @PreAuthorize("@ss.hasPermi('sop:task:remove')")
    @Log(title = "任务步骤快照", businessType = BusinessType.DELETE)
    @DeleteMapping("/{taskStepIds}")
    public AjaxResult remove(@PathVariable Long[] taskStepIds)
    {
        return toAjax(sopTaskStepService.deleteSopTaskStepByTaskStepIds(taskStepIds));
    }
}
