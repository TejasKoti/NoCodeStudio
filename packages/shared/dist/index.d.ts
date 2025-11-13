export type IRNodeKind = "DataSource" | "Transform" | "Model" | "Loss" | "Optimizer" | "TrainingConfig";
export interface IRNode {
    id: string;
    type: IRNodeKind;
    op?: string;
    params?: Record<string, unknown>;
    children?: Array<{
        op: string;
        params?: Record<string, unknown>;
    }>;
    inputs?: string[];
    outputs?: string[];
}
export interface IREdge {
    from: string;
    to: string;
}
export interface IRGraph {
    id: string;
    name: string;
    pytorchVersion: string;
    nodes: IRNode[];
    edges: IREdge[];
}
