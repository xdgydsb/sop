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
import com.ruoyi.system.domain.sop.SopProcess;
import com.ruoyi.system.service.sop.ISopProcessService;

/**
 * SOP process controller
 *
 * @author ruoyi
 */
@RestController
@RequestMapping("/sop/process")
public class SopProcessController extends BaseController
{
    @Autowired
    private ISopProcessService sopProcessService;

    @PreAuthorize("@ss.hasPermi('sop:process:list')")
    @GetMapping("/list")
    public TableDataInfo list(SopProcess sopProcess)
    {
        startPage();
        List<SopProcess> list = sopProcessService.selectSopProcessList(sopProcess);
        return getDataTable(list);
    }

    @Log(title = "SOP流程管理", businessType = BusinessType.EXPORT)
    @PreAuthorize("@ss.hasPermi('sop:process:export')")
    @PostMapping("/export")
    public void export(HttpServletResponse response, SopProcess sopProcess)
    {
        List<SopProcess> list = sopProcessService.selectSopProcessList(sopProcess);
        ExcelUtil<SopProcess> util = new ExcelUtil<SopProcess>(SopProcess.class);
        util.exportExcel(response, list, "SOP流程数据");
    }

    @PreAuthorize("@ss.hasPermi('sop:process:query')")
    @GetMapping(value = "/{sopId}")
    public AjaxResult getInfo(@PathVariable Long sopId)
    {
        return success(sopProcessService.selectSopProcessBySopId(sopId));
    }

    @PreAuthorize("@ss.hasPermi('sop:process:add')")
    @Log(title = "SOP流程管理", businessType = BusinessType.INSERT)
    @PostMapping
    public AjaxResult add(@Validated @RequestBody SopProcess sopProcess)
    {
        sopProcess.setCreateBy(getUsername());
        return toAjax(sopProcessService.insertSopProcess(sopProcess));
    }

    @PreAuthorize("@ss.hasPermi('sop:process:edit')")
    @Log(title = "SOP流程管理", businessType = BusinessType.UPDATE)
    @PutMapping
    public AjaxResult edit(@Validated @RequestBody SopProcess sopProcess)
    {
        sopProcess.setUpdateBy(getUsername());
        return toAjax(sopProcessService.updateSopProcess(sopProcess));
    }

    @PreAuthorize("@ss.hasPermi('sop:process:remove')")
    @Log(title = "SOP流程管理", businessType = BusinessType.DELETE)
    @DeleteMapping("/{sopIds}")
    public AjaxResult remove(@PathVariable Long[] sopIds)
    {
        return toAjax(sopProcessService.deleteSopProcessBySopIds(sopIds));
    }

    @GetMapping("/optionselect")
    public AjaxResult optionselect()
    {
        return success(sopProcessService.selectSopProcessAll());
    }
}
