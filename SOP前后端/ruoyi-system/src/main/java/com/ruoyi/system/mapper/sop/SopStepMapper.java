package com.ruoyi.system.mapper.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopStep;

/**
 * SOP step mapper
 *
 * @author ruoyi
 */
public interface SopStepMapper
{
    public SopStep selectSopStepByStepId(Long stepId);

    public List<SopStep> selectSopStepList(SopStep sopStep);

    public int insertSopStep(SopStep sopStep);

    public int updateSopStep(SopStep sopStep);

    public int deleteSopStepByStepId(Long stepId);

    public int deleteSopStepByStepIds(Long[] stepIds);
}
