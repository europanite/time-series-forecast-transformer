declare module "react-native-web" {
  import type { ComponentType, ReactNode } from "react";

  type WebComponentProps = {
    children?: ReactNode;
    [key: string]: unknown;
  };

  export const View: ComponentType<WebComponentProps>;
  export const Text: ComponentType<WebComponentProps>;
  export const Pressable: ComponentType<WebComponentProps>;
  export const ScrollView: ComponentType<WebComponentProps>;
}
