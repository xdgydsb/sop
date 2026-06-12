package com.ruoyi.system.mapper.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopProcess;

/**
 * SOP process mapper
 *
 * @author ruoyi
 */
public interface SopProcessMapper
{
    public SopProcess selectSopProcessBySopId(Long sopId);

    public List<SopProcess> selectSopProcessList(SopProcess sopProcess);

    public List<SopProcess> selectSopProcessAll();

    public int insertSopProcess(SopProcess sopProcess);

    public int updateSopProcess(SopProcess sopProcess);

    public int deleteSopProcessBySopId(Long sopId);

    public int deleteSopProcessBySopIds(Long[] sopIds);
}
