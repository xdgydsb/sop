package com.ruoyi.system.service.sop.impl;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.ruoyi.system.domain.sop.SopDetectionTask;
import com.ruoyi.system.mapper.sop.SopDetectionTaskMapper;
import com.ruoyi.system.service.sop.ISopDetectionTaskService;

/**
 * SOP detection task service implementation
 *
 * @author ruoyi
 */
@Service
public class SopDetectionTaskServiceImpl implements ISopDetectionTaskService
{
    @Autowired
    private SopDetectionTaskMapper sopDetectionTaskMapper;

    @Override
    public SopDetectionTask selectSopDetectionTaskByTaskId(Long taskId)
    {
        return sopDetectionTaskMapper.selectSopDetectionTaskByTaskId(taskId);
    }

    @Override
    public SopDetectionTask selectSopDetectionTaskByTaskCode(String taskCode)
    {
        return sopDetectionTaskMapper.selectSopDetectionTaskByTaskCode(taskCode);
    }

    @Override
    public List<SopDetectionTask> selectSopDetectionTaskList(SopDetectionTask sopDetectionTask)
    {
        return sopDetectionTaskMapper.selectSopDetectionTaskList(sopDetectionTask);
    }

    @Override
    public int insertSopDetectionTask(SopDetectionTask sopDetectionTask)
    {
        return sopDetectionTaskMapper.insertSopDetectionTask(sopDetectionTask);
    }

    @Override
    public int updateSopDetectionTask(SopDetectionTask sopDetectionTask)
    {
        return sopDetectionTaskMapper.updateSopDetectionTask(sopDetectionTask);
    }

    @Override
    public int deleteSopDetectionTaskByTaskIds(Long[] taskIds)
    {
        return sopDetectionTaskMapper.deleteSopDetectionTaskByTaskIds(taskIds);
    }

    @Override
    public int deleteSopDetectionTaskByTaskId(Long taskId)
    {
        return sopDetectionTaskMapper.deleteSopDetectionTaskByTaskId(taskId);
    }
}
