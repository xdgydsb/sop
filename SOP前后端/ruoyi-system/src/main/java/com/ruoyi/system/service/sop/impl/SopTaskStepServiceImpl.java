package com.ruoyi.system.service.sop.impl;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.ruoyi.system.domain.sop.SopTaskStep;
import com.ruoyi.system.mapper.sop.SopTaskStepMapper;
import com.ruoyi.system.service.sop.ISopTaskStepService;

/**
 * SOP task step service implementation
 *
 * @author ruoyi
 */
@Service
public class SopTaskStepServiceImpl implements ISopTaskStepService
{
    @Autowired
    private SopTaskStepMapper sopTaskStepMapper;

    @Override
    public SopTaskStep selectSopTaskStepByTaskStepId(Long taskStepId)
    {
        return sopTaskStepMapper.selectSopTaskStepByTaskStepId(taskStepId);
    }

    @Override
    public List<SopTaskStep> selectSopTaskStepList(SopTaskStep sopTaskStep)
    {
        return sopTaskStepMapper.selectSopTaskStepList(sopTaskStep);
    }

    @Override
    public int insertSopTaskStep(SopTaskStep sopTaskStep)
    {
        return sopTaskStepMapper.insertSopTaskStep(sopTaskStep);
    }

    @Override
    public int updateSopTaskStep(SopTaskStep sopTaskStep)
    {
        return sopTaskStepMapper.updateSopTaskStep(sopTaskStep);
    }

    @Override
    public int deleteSopTaskStepByTaskStepIds(Long[] taskStepIds)
    {
        return sopTaskStepMapper.deleteSopTaskStepByTaskStepIds(taskStepIds);
    }

    @Override
    public int deleteSopTaskStepByTaskStepId(Long taskStepId)
    {
        return sopTaskStepMapper.deleteSopTaskStepByTaskStepId(taskStepId);
    }
}
