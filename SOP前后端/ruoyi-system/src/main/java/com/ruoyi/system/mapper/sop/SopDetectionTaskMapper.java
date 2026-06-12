package com.ruoyi.system.mapper.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopDetectionTask;

/**
 * SOP detection task mapper
 *
 * @author ruoyi
 */
public interface SopDetectionTaskMapper
{
    public SopDetectionTask selectSopDetectionTaskByTaskId(Long taskId);

    public SopDetectionTask selectSopDetectionTaskByTaskCode(String taskCode);

    public List<SopDetectionTask> selectSopDetectionTaskList(SopDetectionTask sopDetectionTask);

    public int insertSopDetectionTask(SopDetectionTask sopDetectionTask);

    public int updateSopDetectionTask(SopDetectionTask sopDetectionTask);

    public int deleteSopDetectionTaskByTaskId(Long taskId);

    public int deleteSopDetectionTaskByTaskIds(Long[] taskIds);
}
