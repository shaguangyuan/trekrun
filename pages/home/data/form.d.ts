declare const form: {
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
declare const skylineForm: {
    name: string;
    icon: string;
    childArr: {
        name: string;
        label: string;
    }[];
};
export { form, skylineForm };
