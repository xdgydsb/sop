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
import com.ruoyi.system.domain.sop.SopAlarmRecord;
import com.ruoyi.system.service.sop.ISopAlarmRecordService;

/**
 * SOP alarm record controller
 *
 * @author ruoyi
 */
@RestController
@RequestMapping("/sop/alarm")
public class SopAlarmRecordController extends BaseController
{
    @Autowired
    private ISopAlarmRecordService sopAlarmRecordService;

    @PreAuthorize("@ss.hasPermi('sop:alarm:list')")
    @GetMapping("/list")
    public TableDataInfo list(SopAlarmRecord sopAlarmRecord)
    {
        startPage();
        List<SopAlarmRecord> list = sopAlarmRecordService.selectSopAlarmRecordList(sopAlarmRecord);
        return getDataTable(list);
    }

    @Log(title = "告警记录", businessType = BusinessType.EXPORT)
    @PreAuthorize("@ss.hasPermi('sop:alarm:export')")
    @PostMapping("/export")
    public void export(HttpServletResponse response, SopAlarmRecord sopAlarmRecord)
    {
        List<SopAlarmRecord> list = sopAlarmRecordService.selectSopAlarmRecordList(sopAlarmRecord);
        ExcelUtil<SopAlarmRecord> util = new ExcelUtil<SopAlarmRecord>(SopAlarmRecord.class);
        util.exportExcel(response, list, "告警记录数据");
    }

    @PreAuthorize("@ss.hasPermi('sop:alarm:query')")
    @GetMapping(value = "/{alarmId}")
    public AjaxResult getInfo(@PathVariable Long alarmId)
    {
        return success(sopAlarmRecordService.selectSopAlarmRecordByAlarmId(alarmId));
    }

    @PreAuthorize("@ss.hasPermi('sop:alarm:add')")
    @Log(title = "告警记录", businessType = BusinessType.INSERT)
    @PostMapping
    public AjaxResult add(@Validated @RequestBody SopAlarmRecord sopAlarmRecord)
    {
        sopAlarmRecord.setCreateBy(getUsername());
        return toAjax(sopAlarmRecordService.insertSopAlarmRecord(sopAlarmRecord));
    }

    @PreAuthorize("@ss.hasPermi('sop:alarm:edit')")
    @Log(title = "告警记录", businessType = BusinessType.UPDATE)
    @PutMapping
    public AjaxResult edit(@Validated @RequestBody SopAlarmRecord sopAlarmRecord)
    {
        sopAlarmRecord.setUpdateBy(getUsername());
        return toAjax(sopAlarmRecordService.updateSopAlarmRecord(sopAlarmRecord));
    }

    @PreAuthorize("@ss.hasPermi('sop:alarm:remove')")
    @Log(title = "告警记录", businessType = BusinessType.DELETE)
    @DeleteMapping("/{alarmIds}")
    public AjaxResult remove(@PathVariable Long[] alarmIds)
    {
        return toAjax(sopAlarmRecordService.deleteSopAlarmRecordByAlarmIds(alarmIds));
    }
}
