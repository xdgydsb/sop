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
import com.ruoyi.common.annotation.Anonymous;
import com.ruoyi.common.annotation.Log;
import com.ruoyi.common.core.controller.BaseController;
import com.ruoyi.common.core.domain.AjaxResult;
import com.ruoyi.common.core.page.TableDataInfo;
import com.ruoyi.common.enums.BusinessType;
import com.ruoyi.common.utils.poi.ExcelUtil;
import com.ruoyi.system.domain.sop.SopDetectionEvent;
import com.ruoyi.system.service.sop.ISopDetectionEventService;

/**
 * SOP detection event log controller
 *
 * @author ruoyi
 */
@RestController
@RequestMapping("/sop/event")
public class SopDetectionEventController extends BaseController
{
    @Autowired
    private ISopDetectionEventService sopDetectionEventService;

    @PreAuthorize("@ss.hasPermi('sop:event:list')")
    @GetMapping("/list")
    public TableDataInfo list(SopDetectionEvent sopDetectionEvent)
    {
        startPage();
        List<SopDetectionEvent> list = sopDetectionEventService.selectSopDetectionEventList(sopDetectionEvent);
        return getDataTable(list);
    }

    /**
     * Receive visual detection event from external vision system.
     */
    @Anonymous
    @PostMapping("/receive")
    public AjaxResult receive(@RequestBody SopDetectionEvent sopDetectionEvent)
    {
        if (sopDetectionEvent == null || sopDetectionEvent.getTaskCode() == null || sopDetectionEvent.getTaskCode().trim().isEmpty())
        {
            return error("任务编码不能为空");
        }
        if (sopDetectionEvent.getEventCode() == null || sopDetectionEvent.getEventCode().trim().isEmpty())
        {
            return error("事件编码不能为空");
        }
        return success(sopDetectionEventService.receiveVisualEvent(sopDetectionEvent));
    }

    @Log(title = "检测事件日志", businessType = BusinessType.EXPORT)
    @PreAuthorize("@ss.hasPermi('sop:event:export')")
    @PostMapping("/export")
    public void export(HttpServletResponse response, SopDetectionEvent sopDetectionEvent)
    {
        List<SopDetectionEvent> list = sopDetectionEventService.selectSopDetectionEventList(sopDetectionEvent);
        ExcelUtil<SopDetectionEvent> util = new ExcelUtil<SopDetectionEvent>(SopDetectionEvent.class);
        util.exportExcel(response, list, "检测事件日志数据");
    }

    @PreAuthorize("@ss.hasPermi('sop:event:query')")
    @GetMapping(value = "/{eventLogId}")
    public AjaxResult getInfo(@PathVariable Long eventLogId)
    {
        return success(sopDetectionEventService.selectSopDetectionEventByEventLogId(eventLogId));
    }

    @PreAuthorize("@ss.hasPermi('sop:event:add')")
    @Log(title = "检测事件日志", businessType = BusinessType.INSERT)
    @PostMapping
    public AjaxResult add(@Validated @RequestBody SopDetectionEvent sopDetectionEvent)
    {
        sopDetectionEvent.setCreateBy(getUsername());
        return toAjax(sopDetectionEventService.insertSopDetectionEvent(sopDetectionEvent));
    }

    @PreAuthorize("@ss.hasPermi('sop:event:edit')")
    @Log(title = "检测事件日志", businessType = BusinessType.UPDATE)
    @PutMapping
    public AjaxResult edit(@Validated @RequestBody SopDetectionEvent sopDetectionEvent)
    {
        sopDetectionEvent.setUpdateBy(getUsername());
        return toAjax(sopDetectionEventService.updateSopDetectionEvent(sopDetectionEvent));
    }

    @PreAuthorize("@ss.hasPermi('sop:event:remove')")
    @Log(title = "检测事件日志", businessType = BusinessType.DELETE)
    @DeleteMapping("/{eventLogIds}")
    public AjaxResult remove(@PathVariable Long[] eventLogIds)
    {
        return toAjax(sopDetectionEventService.deleteSopDetectionEventByEventLogIds(eventLogIds));
    }
}
