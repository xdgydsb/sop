package com.ruoyi.system.service.sop.impl;

import java.util.List;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import com.ruoyi.system.domain.sop.SopProcess;
import com.ruoyi.system.mapper.sop.SopProcessMapper;
import com.ruoyi.system.service.sop.ISopProcessService;

/**
 * SOP process service implementation
 *
 * @author ruoyi
 */
@Service
public class SopProcessServiceImpl implements ISopProcessService
{
    @Autowired
    private SopProcessMapper sopProcessMapper;

    @Override
    public SopProcess selectSopProcessBySopId(Long sopId)
    {
        return sopProcessMapper.selectSopProcessBySopId(sopId);
    }

    @Override
    public List<SopProcess> selectSopProcessList(SopProcess sopProcess)
    {
        return sopProcessMapper.selectSopProcessList(sopProcess);
    }

    @Override
    public List<SopProcess> selectSopProcessAll()
    {
        return sopProcessMapper.selectSopProcessAll();
    }

    @Override
    public int insertSopProcess(SopProcess sopProcess)
    {
        return sopProcessMapper.insertSopProcess(sopProcess);
    }

    @Override
    public int updateSopProcess(SopProcess sopProcess)
    {
        return sopProcessMapper.updateSopProcess(sopProcess);
    }

    @Override
    public int deleteSopProcessBySopIds(Long[] sopIds)
    {
        return sopProcessMapper.deleteSopProcessBySopIds(sopIds);
    }

    @Override
    public int deleteSopProcessBySopId(Long sopId)
    {
        return sopProcessMapper.deleteSopProcessBySopId(sopId);
    }
}
