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
import com.ruoyi.system.domain.sop.SopDetectionTask;
import com.ruoyi.system.service.sop.ISopDetectionTaskService;

/**
 * SOP detection task controller
 *
 * @author ruoyi
 */
@RestController
@RequestMapping("/sop/task")
public class SopDetectionTaskController extends BaseController
{
    @Autowired
    private ISopDetectionTaskService sopDetectionTaskService;

    @PreAuthorize("@ss.hasPermi('sop:task:list')")
    @GetMapping("/list")
    public TableDataInfo list(SopDetectionTask sopDetectionTask)
    {
        startPage();
        List<SopDetectionTask> list = sopDetectionTaskService.selectSopDetectionTaskList(sopDetectionTask);
        return getDataTable(list);
    }

    @Log(title = "检测任务管理", businessType = BusinessType.EXPORT)
    @PreAuthorize("@ss.hasPermi('sop:task:export')")
    @PostMapping("/export")
    public void export(HttpServletResponse response, SopDetectionTask sopDetectionTask)
    {
        List<SopDetectionTask> list = sopDetectionTaskService.selectSopDetectionTaskList(sopDetectionTask);
        ExcelUtil<SopDetectionTask> util = new ExcelUtil<SopDetectionTask>(SopDetectionTask.class);
        util.exportExcel(response, list, "检测任务数据");
    }

    @PreAuthorize("@ss.hasPermi('sop:task:query')")
    @GetMapping(value = "/{taskId}")
    public AjaxResult getInfo(@PathVariable Long taskId)
    {
        return success(sopDetectionTaskService.selectSopDetectionTaskByTaskId(taskId));
    }

    @PreAuthorize("@ss.hasPermi('sop:task:add')")
    @Log(title = "检测任务管理", businessType = BusinessType.INSERT)
    @PostMapping
    public AjaxResult add(@Validated @RequestBody SopDetectionTask sopDetectionTask)
    {
        sopDetectionTask.setCreateBy(getUsername());
        return toAjax(sopDetectionTaskService.insertSopDetectionTask(sopDetectionTask));
    }

    @PreAuthorize("@ss.hasPermi('sop:task:edit')")
    @Log(title = "检测任务管理", businessType = BusinessType.UPDATE)
    @PutMapping
    public AjaxResult edit(@Validated @RequestBody SopDetectionTask sopDetectionTask)
    {
        sopDetectionTask.setUpdateBy(getUsername());
        return toAjax(sopDetectionTaskService.updateSopDetectionTask(sopDetectionTask));
    }

    @PreAuthorize("@ss.hasPermi('sop:task:remove')")
    @Log(title = "检测任务管理", businessType = BusinessType.DELETE)
    @DeleteMapping("/{taskIds}")
    public AjaxResult remove(@PathVariable Long[] taskIds)
    {
        return toAjax(sopDetectionTaskService.deleteSopDetectionTaskByTaskIds(taskIds));
    }
}
