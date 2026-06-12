package com.ruoyi.system.service.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopStep;

/**
 * SOP step service
 *
 * @author ruoyi
 */
public interface ISopStepService
{
    public SopStep selectSopStepByStepId(Long stepId);

    public List<SopStep> selectSopStepList(SopStep sopStep);

    public int insertSopStep(SopStep sopStep);

    public int updateSopStep(SopStep sopStep);

    public int deleteSopStepByStepIds(Long[] stepIds);

    public int deleteSopStepByStepId(Long stepId);
}
