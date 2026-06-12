package com.ruoyi.system.service.sop.impl;

import java.math.BigDecimal;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;
import com.alibaba.fastjson2.JSON;
import org.apache.commons.lang3.StringUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;
import com.ruoyi.system.domain.sop.SopAlarmRecord;
import com.ruoyi.system.domain.sop.SopDetectionEvent;
import com.ruoyi.system.domain.sop.SopDetectionTask;
import com.ruoyi.system.domain.sop.SopStep;
import com.ruoyi.system.domain.sop.SopTaskStep;
import com.ruoyi.system.mapper.sop.SopAlarmRecordMapper;
import com.ruoyi.system.mapper.sop.SopDetectionEventMapper;
import com.ruoyi.system.mapper.sop.SopDetectionTaskMapper;
import com.ruoyi.system.mapper.sop.SopStepMapper;
import com.ruoyi.system.mapper.sop.SopTaskStepMapper;
import com.ruoyi.system.service.sop.ISopDetectionEventService;

/**
 * SOP detection event service implementation
 *
 * @author ruoyi
 */
@Service
public class SopDetectionEventServiceImpl implements ISopDetectionEventService
{
    @Autowired
    private SopDetectionEventMapper sopDetectionEventMapper;

    @Autowired
    private SopDetectionTaskMapper sopDetectionTaskMapper;

    @Autowired
    private SopStepMapper sopStepMapper;

    @Autowired
    private SopTaskStepMapper sopTaskStepMapper;

    @Autowired
    private SopAlarmRecordMapper sopAlarmRecordMapper;

    @Override
    public SopDetectionEvent selectSopDetectionEventByEventLogId(Long eventLogId)
    {
        return sopDetectionEventMapper.selectSopDetectionEventByEventLogId(eventLogId);
    }

    @Override
    public List<SopDetectionEvent> selectSopDetectionEventList(SopDetectionEvent sopDetectionEvent)
    {
        return sopDetectionEventMapper.selectSopDetectionEventList(sopDetectionEvent);
    }

    @Override
    @Transactional
    public SopDetectionEvent receiveVisualEvent(SopDetectionEvent sopDetectionEvent)
    {
        Date now = new Date();
        if (sopDetectionEvent.getReceiveTime() == null)
        {
            sopDetectionEvent.setReceiveTime(now);
        }
        if (sopDetectionEvent.getEventTime() == null)
        {
            sopDetectionEvent.setEventTime(now);
        }
        if (StringUtils.isBlank(sopDetectionEvent.getRawPayload()))
        {
            sopDetectionEvent.setRawPayload(JSON.toJSONString(sopDetectionEvent));
        }
        sopDetectionEvent.setCreateBy("vision");

        SopDetectionTask task = sopDetectionTaskMapper.selectSopDetectionTaskByTaskCode(sopDetectionEvent.getTaskCode());
        if (task == null)
        {
            sopDetectionEvent.setJudgeResult("UNKNOWN");
            sopDetectionEvent.setJudgeMessage("检测任务不存在：" + sopDetectionEvent.getTaskCode());
            sopDetectionEventMapper.insertSopDetectionEvent(sopDetectionEvent);
            insertAlarm(sopDetectionEvent, null, "TASK_NOT_FOUND", "ERROR", sopDetectionEvent.getJudgeMessage());
            return sopDetectionEvent;
        }

        fillEventFromTask(sopDetectionEvent, task);
        List<SopTaskStep> taskSteps = ensureTaskSteps(task);
        if (taskSteps.isEmpty())
        {
            sopDetectionEvent.setJudgeResult("UNKNOWN");
            sopDetectionEvent.setJudgeMessage("任务未配置SOP步骤");
            sopDetectionEventMapper.insertSopDetectionEvent(sopDetectionEvent);
            insertAlarm(sopDetectionEvent, task, "STEP_NOT_CONFIGURED", "ERROR", sopDetectionEvent.getJudgeMessage());
            return sopDetectionEvent;
        }

        if (isClosedTask(task))
        {
            sopDetectionEvent.setJudgeResult("IGNORED");
            sopDetectionEvent.setJudgeMessage("任务已结束，事件已记录但未参与判定");
            sopDetectionEventMapper.insertSopDetectionEvent(sopDetectionEvent);
            insertAlarm(sopDetectionEvent, task, "TASK_CLOSED", "WARN", sopDetectionEvent.getJudgeMessage());
            return sopDetectionEvent;
        }

        SopTaskStep currentStep = findCurrentStep(task, taskSteps);
        if (currentStep == null)
        {
            sopDetectionEvent.setJudgeResult("IGNORED");
            sopDetectionEvent.setJudgeMessage("任务没有待检测步骤，事件已记录但未参与判定");
            sopDetectionEventMapper.insertSopDetectionEvent(sopDetectionEvent);
            insertAlarm(sopDetectionEvent, task, "NO_PENDING_STEP", "WARN", sopDetectionEvent.getJudgeMessage());
            return sopDetectionEvent;
        }

        sopDetectionEvent.setStepId(currentStep.getStepId());
        sopDetectionEvent.setStepNo(currentStep.getStepNo());
        if (isClosedTask(task))
        {
            sopDetectionEvent.setJudgeResult("IGNORED");
            sopDetectionEvent.setJudgeMessage("任务已结束，事件已记录但未参与判定");
            sopDetectionEventMapper.insertSopDetectionEvent(sopDetectionEvent);
            insertAlarm(sopDetectionEvent, task, "TASK_CLOSED", "WARN", sopDetectionEvent.getJudgeMessage());
            return sopDetectionEvent;
        }

        if (!StringUtils.equals(currentStep.getExpectedEvent(), sopDetectionEvent.getEventCode()))
        {
            sopDetectionEvent.setJudgeResult("FAIL");
            sopDetectionEvent.setJudgeMessage("事件不匹配，当前步骤期望事件：" + currentStep.getExpectedEvent());
            sopDetectionEventMapper.insertSopDetectionEvent(sopDetectionEvent);
            insertAlarm(sopDetectionEvent, task, "EVENT_MISMATCH", "WARN", sopDetectionEvent.getJudgeMessage());
            markTaskRunning(task, now);
            return sopDetectionEvent;
        }

        BigDecimal confidence = sopDetectionEvent.getConfidence();
        BigDecimal requiredConfidence = currentStep.getRequiredConfidence();
        if (confidence != null && requiredConfidence != null && confidence.compareTo(requiredConfidence) < 0)
        {
            sopDetectionEvent.setJudgeResult("FAIL");
            sopDetectionEvent.setJudgeMessage("置信度不足，当前置信度：" + confidence + "，最低要求：" + requiredConfidence);
            sopDetectionEventMapper.insertSopDetectionEvent(sopDetectionEvent);
            insertAlarm(sopDetectionEvent, task, "LOW_CONFIDENCE", "WARN", sopDetectionEvent.getJudgeMessage());
            markTaskRunning(task, now);
            return sopDetectionEvent;
        }

        sopDetectionEvent.setJudgeResult("PASS");
        sopDetectionEvent.setJudgeMessage("当前步骤检测通过");
        sopDetectionEventMapper.insertSopDetectionEvent(sopDetectionEvent);
        passCurrentStep(currentStep, sopDetectionEvent, now);
        advanceTask(task, taskSteps, currentStep, now);
        return sopDetectionEvent;
    }

    @Override
    public int insertSopDetectionEvent(SopDetectionEvent sopDetectionEvent)
    {
        return sopDetectionEventMapper.insertSopDetectionEvent(sopDetectionEvent);
    }

    @Override
    public int updateSopDetectionEvent(SopDetectionEvent sopDetectionEvent)
    {
        return sopDetectionEventMapper.updateSopDetectionEvent(sopDetectionEvent);
    }

    @Override
    public int deleteSopDetectionEventByEventLogIds(Long[] eventLogIds)
    {
        return sopDetectionEventMapper.deleteSopDetectionEventByEventLogIds(eventLogIds);
    }

    @Override
    public int deleteSopDetectionEventByEventLogId(Long eventLogId)
    {
        return sopDetectionEventMapper.deleteSopDetectionEventByEventLogId(eventLogId);
    }

    private void fillEventFromTask(SopDetectionEvent event, SopDetectionTask task)
    {
        event.setTaskId(task.getTaskId());
        if (StringUtils.isBlank(event.getProductCode()))
        {
            event.setProductCode(task.getProductCode());
        }
        if (StringUtils.isBlank(event.getStationCode()))
        {
            event.setStationCode(task.getStationCode());
        }
        if (StringUtils.isBlank(event.getCameraCode()))
        {
            event.setCameraCode(task.getCameraCode());
        }
    }

    private List<SopTaskStep> ensureTaskSteps(SopDetectionTask task)
    {
        SopTaskStep query = new SopTaskStep();
        query.setTaskId(task.getTaskId());
        List<SopTaskStep> taskSteps = sopTaskStepMapper.selectSopTaskStepList(query);
        if (!taskSteps.isEmpty())
        {
            return taskSteps;
        }

        SopStep stepQuery = new SopStep();
        stepQuery.setSopId(task.getSopId());
        stepQuery.setStatus("0");
        List<SopStep> steps = sopStepMapper.selectSopStepList(stepQuery);
        for (SopStep step : steps)
        {
            SopTaskStep taskStep = new SopTaskStep();
            taskStep.setTaskId(task.getTaskId());
            taskStep.setStepId(step.getStepId());
            taskStep.setStepNo(step.getStepNo());
            taskStep.setStepName(step.getStepName());
            taskStep.setExpectedEvent(step.getExpectedEvent());
            taskStep.setRequiredConfidence(step.getRequiredConfidence());
            taskStep.setStepStatus("PENDING");
            taskStep.setCreateBy("vision");
            sopTaskStepMapper.insertSopTaskStep(taskStep);
        }
        return sopTaskStepMapper.selectSopTaskStepList(query);
    }

    private SopTaskStep findCurrentStep(SopDetectionTask task, List<SopTaskStep> taskSteps)
    {
        Integer currentStepNo = task.getCurrentStepNo() == null ? 1 : task.getCurrentStepNo();
        SopTaskStep firstPending = null;
        for (SopTaskStep step : taskSteps)
        {
            if ("PENDING".equals(step.getStepStatus()) && firstPending == null)
            {
                firstPending = step;
            }
            if (currentStepNo.equals(step.getStepNo()) && !"PASSED".equals(step.getStepStatus()))
            {
                return step;
            }
        }
        return firstPending;
    }

    private void passCurrentStep(SopTaskStep currentStep, SopDetectionEvent event, Date now)
    {
        currentStep.setStepStatus("PASSED");
        currentStep.setPassTime(now);
        currentStep.setEventLogId(event.getEventLogId());
        currentStep.setUpdateBy("vision");
        sopTaskStepMapper.updateSopTaskStep(currentStep);
    }

    private void advanceTask(SopDetectionTask task, List<SopTaskStep> taskSteps, SopTaskStep currentStep, Date now)
    {
        Integer nextStepNo = null;
        for (SopTaskStep step : taskSteps)
        {
            if (step.getStepNo() > currentStep.getStepNo() && !"PASSED".equals(step.getStepStatus()))
            {
                nextStepNo = step.getStepNo();
                break;
            }
        }

        task.setUpdateBy("vision");
        if (task.getStartTime() == null)
        {
            task.setStartTime(now);
        }
        if (nextStepNo == null)
        {
            task.setTaskStatus("PASSED");
            task.setEndTime(now);
        }
        else
        {
            task.setTaskStatus("RUNNING");
            task.setCurrentStepNo(nextStepNo);
        }
        sopDetectionTaskMapper.updateSopDetectionTask(task);
    }

    private void markTaskRunning(SopDetectionTask task, Date now)
    {
        if (!isClosedTask(task) && !"RUNNING".equals(task.getTaskStatus()))
        {
            task.setTaskStatus("RUNNING");
            task.setUpdateBy("vision");
            if (task.getStartTime() == null)
            {
                task.setStartTime(now);
            }
            sopDetectionTaskMapper.updateSopDetectionTask(task);
        }
    }

    private boolean isClosedTask(SopDetectionTask task)
    {
        return "PASSED".equals(task.getTaskStatus()) || "FINISHED".equals(task.getTaskStatus()) || "CANCELLED".equals(task.getTaskStatus());
    }

    private void insertAlarm(SopDetectionEvent event, SopDetectionTask task, String alarmType, String alarmLevel, String message)
    {
        SopAlarmRecord alarm = new SopAlarmRecord();
        alarm.setAlarmCode(nextAlarmCode());
        alarm.setTaskId(task == null ? event.getTaskId() : task.getTaskId());
        alarm.setTaskCode(event.getTaskCode());
        alarm.setProductCode(event.getProductCode());
        alarm.setStationCode(event.getStationCode());
        alarm.setCameraCode(event.getCameraCode());
        alarm.setAlarmType(alarmType);
        alarm.setAlarmLevel(alarmLevel);
        alarm.setAlarmMessage(message);
        alarm.setEventLogId(event.getEventLogId());
        alarm.setEventCode(event.getEventCode());
        alarm.setEventName(event.getEventName());
        alarm.setStepId(event.getStepId());
        alarm.setStepNo(event.getStepNo());
        alarm.setAlarmTime(event.getReceiveTime() == null ? new Date() : event.getReceiveTime());
        alarm.setHandleStatus("UNHANDLED");
        alarm.setCreateBy("vision");
        sopAlarmRecordMapper.insertSopAlarmRecord(alarm);
    }

    private String nextAlarmCode()
    {
        return "ALM" + new SimpleDateFormat("yyyyMMddHHmmssSSS").format(new Date())
            + ThreadLocalRandom.current().nextInt(1000, 10000);
    }
}
