package com.ruoyi.system.service.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopProcess;

/**
 * SOP process service
 *
 * @author ruoyi
 */
public interface ISopProcessService
{
    public SopProcess selectSopProcessBySopId(Long sopId);

    public List<SopProcess> selectSopProcessList(SopProcess sopProcess);

    public List<SopProcess> selectSopProcessAll();

    public int insertSopProcess(SopProcess sopProcess);

    public int updateSopProcess(SopProcess sopProcess);

    public int deleteSopProcessBySopIds(Long[] sopIds);

    public int deleteSopProcessBySopId(Long sopId);
}
