package com.ruoyi.system.service.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopTaskStep;

/**
 * SOP task step service
 *
 * @author ruoyi
 */
public interface ISopTaskStepService
{
    public SopTaskStep selectSopTaskStepByTaskStepId(Long taskStepId);

    public List<SopTaskStep> selectSopTaskStepList(SopTaskStep sopTaskStep);

    public int insertSopTaskStep(SopTaskStep sopTaskStep);

    public int updateSopTaskStep(SopTaskStep sopTaskStep);

    public int deleteSopTaskStepByTaskStepIds(Long[] taskStepIds);

    public int deleteSopTaskStepByTaskStepId(Long taskStepId);
}
