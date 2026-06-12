package com.ruoyi.system.service.sop;

import java.util.List;
import com.ruoyi.system.domain.sop.SopDetectionTask;

/**
 * SOP detection task service
 *
 * @author ruoyi
 */
public interface ISopDetectionTaskService
{
    public SopDetectionTask selectSopDetectionTaskByTaskId(Long taskId);

    public SopDetectionTask selectSopDetectionTaskByTaskCode(String taskCode);

    public List<SopDetectionTask> selectSopDetectionTaskList(SopDetectionTask sopDetectionTask);

    public int insertSopDetectionTask(SopDetectionTask sopDetectionTask);

    public int updateSopDetectionTask(SopDetectionTask sopDetectionTask);

    public int deleteSopDetectionTaskByTaskIds(Long[] taskIds);

    public int deleteSopDetectionTaskByTaskId(Long taskId);
}
