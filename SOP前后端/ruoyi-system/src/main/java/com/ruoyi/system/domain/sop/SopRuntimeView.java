package com.ruoyi.system.domain.sop;

import java.util.List;

public class SopRuntimeView
{
    private SopDetectionTask task;
    private List<SopTaskStep> steps;
    private List<SopDetectionEvent> recentEvents;
    private List<SopAlarmRecord> recentAlarms;

    public SopDetectionTask getTask()
    {
        return task;
    }

    public void setTask(SopDetectionTask task)
    {
        this.task = task;
    }

    public List<SopTaskStep> getSteps()
    {
        return steps;
    }

    public void setSteps(List<SopTaskStep> steps)
    {
        this.steps = steps;
    }

    public List<SopDetectionEvent> getRecentEvents()
    {
        return recentEvents;
    }

    public void setRecentEvents(List<SopDetectionEvent> recentEvents)
    {
        this.recentEvents = recentEvents;
    }

    public List<SopAlarmRecord> getRecentAlarms()
    {
        return recentAlarms;
    }

    public void setRecentAlarms(List<SopAlarmRecord> recentAlarms)
    {
        this.recentAlarms = recentAlarms;
    }
}
