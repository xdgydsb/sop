package com.ruoyi.system.service.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopAlarmRecord;

/**
 * SOP alarm record service
 *
 * @author ruoyi
 */
public interface ISopAlarmRecordService
{
    public SopAlarmRecord selectSopAlarmRecordByAlarmId(Long alarmId);

    public List<SopAlarmRecord> selectSopAlarmRecordList(SopAlarmRecord sopAlarmRecord);

    public int insertSopAlarmRecord(SopAlarmRecord sopAlarmRecord);

    public int updateSopAlarmRecord(SopAlarmRecord sopAlarmRecord);

    public int deleteSopAlarmRecordByAlarmIds(Long[] alarmIds);

    public int deleteSopAlarmRecordByAlarmId(Long alarmId);
}
