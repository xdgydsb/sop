package com.ruoyi.system.service.sop.impl;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.ruoyi.system.domain.sop.SopStep;
import com.ruoyi.system.mapper.sop.SopStepMapper;
import com.ruoyi.system.service.sop.ISopStepService;

/**
 * SOP step service implementation
 *
 * @author ruoyi
 */
@Service
public class SopStepServiceImpl implements ISopStepService
{
    @Autowired
    private SopStepMapper sopStepMapper;

    @Override
    public SopStep selectSopStepByStepId(Long stepId)
    {
        return sopStepMapper.selectSopStepByStepId(stepId);
    }

    @Override
    public List<SopStep> selectSopStepList(SopStep sopStep)
    {
        return sopStepMapper.selectSopStepList(sopStep);
    }

    @Override
    public int insertSopStep(SopStep sopStep)
    {
        return sopStepMapper.insertSopStep(sopStep);
    }

    @Override
    public int updateSopStep(SopStep sopStep)
    {
        return sopStepMapper.updateSopStep(sopStep);
    }

    @Override
    public int deleteSopStepByStepIds(Long[] stepIds)
    {
        return sopStepMapper.deleteSopStepByStepIds(stepIds);
    }

    @Override
    public int deleteSopStepByStepId(Long stepId)
    {
        return sopStepMapper.deleteSopStepByStepId(stepId);
    }
}
