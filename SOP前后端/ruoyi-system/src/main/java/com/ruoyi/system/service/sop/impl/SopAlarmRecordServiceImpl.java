package com.ruoyi.system.service.sop.impl;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.ruoyi.system.domain.sop.SopAlarmRecord;
import com.ruoyi.system.mapper.sop.SopAlarmRecordMapper;
import com.ruoyi.system.service.sop.ISopAlarmRecordService;

/**
 * SOP alarm record service implementation
 *
 * @author ruoyi
 */
@Service
public class SopAlarmRecordServiceImpl implements ISopAlarmRecordService
{
    @Autowired
    private SopAlarmRecordMapper sopAlarmRecordMapper;

    @Override
    public SopAlarmRecord selectSopAlarmRecordByAlarmId(Long alarmId)
    {
        return sopAlarmRecordMapper.selectSopAlarmRecordByAlarmId(alarmId);
    }

    @Override
    public List<SopAlarmRecord> selectSopAlarmRecordList(SopAlarmRecord sopAlarmRecord)
    {
        return sopAlarmRecordMapper.selectSopAlarmRecordList(sopAlarmRecord);
    }

    @Override
    public int insertSopAlarmRecord(SopAlarmRecord sopAlarmRecord)
    {
        return sopAlarmRecordMapper.insertSopAlarmRecord(sopAlarmRecord);
    }

    @Override
    public int updateSopAlarmRecord(SopAlarmRecord sopAlarmRecord)
    {
        return sopAlarmRecordMapper.updateSopAlarmRecord(sopAlarmRecord);
    }

    @Override
    public int deleteSopAlarmRecordByAlarmIds(Long[] alarmIds)
    {
        return sopAlarmRecordMapper.deleteSopAlarmRecordByAlarmIds(alarmIds);
    }

    @Override
    public int deleteSopAlarmRecordByAlarmId(Long alarmId)
    {
        return sopAlarmRecordMapper.deleteSopAlarmRecordByAlarmId(alarmId);
    }
}
