package com.ruoyi.system.mapper.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopTaskStep;

/**
 * SOP task step mapper
 *
 * @author ruoyi
 */
public interface SopTaskStepMapper
{
    public SopTaskStep selectSopTaskStepByTaskStepId(Long taskStepId);

    public List<SopTaskStep> selectSopTaskStepList(SopTaskStep sopTaskStep);

    public int insertSopTaskStep(SopTaskStep sopTaskStep);

    public int updateSopTaskStep(SopTaskStep sopTaskStep);

    public int deleteSopTaskStepByTaskStepId(Long taskStepId);

    public int deleteSopTaskStepByTaskStepIds(Long[] taskStepIds);
}
