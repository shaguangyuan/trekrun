declare const base: {
    name: string;
    icon: string;
    childArr: ({
        name: string;
        label: string;
        path?: undefined;
    } | {
        name: string;
        label: string;
        path: string;
    })[];
};
declare const skylineBase: {
    name: string;
    icon: string;
    childArr: ({
        name: string;
        label: string;
        path?: undefined;
    } | {
        name: string;
        label: string;
        path: string;
    })[];
};
export { base, skylineBase };
