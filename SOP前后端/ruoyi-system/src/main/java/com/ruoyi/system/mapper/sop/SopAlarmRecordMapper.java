package com.ruoyi.system.mapper.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopAlarmRecord;

/**
 * SOP alarm record mapper
 *
 * @author ruoyi
 */
public interface SopAlarmRecordMapper
{
    public SopAlarmRecord selectSopAlarmRecordByAlarmId(Long alarmId);

    public List<SopAlarmRecord> selectSopAlarmRecordList(SopAlarmRecord sopAlarmRecord);

    public int insertSopAlarmRecord(SopAlarmRecord sopAlarmRecord);

    public int updateSopAlarmRecord(SopAlarmRecord sopAlarmRecord);

    public int deleteSopAlarmRecordByAlarmId(Long alarmId);

    public int deleteSopAlarmRecordByAlarmIds(Long[] alarmIds);
}
