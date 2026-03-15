export type BridgeCommandResult<T = unknown> = {
    ok: boolean;
    code: number | null;
    stdout: string;
    stderr: string;
    payload: T;
};
export type NemoBridgeHealth = {
    ok: boolean;
    natAvailable: boolean;
    natExecutable: string | null;
    python: string;
    pythonVersion: string;
    platform: string;
    cwd: string;
};
export type NemoWorkflowRunResult = {
    ok: boolean;
    code: number;
    command: string[];
    cwd: string;
    stdout: string;
    stderr: string;
};
export type NemoBridgeOptions = {
    rootDir?: string;
    pythonBinary?: string;
    natWorkdir?: string;
    timeoutMs?: number;
    env?: NodeJS.ProcessEnv;
};
export type RunWorkflowOptions = {
    configFile: string;
    input: string;
    natWorkdir?: string;
    timeoutMs?: number;
    extraArgs?: string[];
};
export declare class NemoBridge {
    readonly rootDir: string;
    readonly pythonBinary: string;
    readonly natWorkdir: string;
    readonly timeoutMs: number;
    readonly env: NodeJS.ProcessEnv;
    constructor(options?: NemoBridgeOptions);
    health(): Promise<BridgeCommandResult<NemoBridgeHealth>>;
    runWorkflow(options: RunWorkflowOptions): Promise<BridgeCommandResult<NemoWorkflowRunResult>>;
    private runBridgeCommand;
}
